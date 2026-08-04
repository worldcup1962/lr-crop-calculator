--[[
ApplyCropFromCSV.lua

crop_calculator.py が出力した CSV (filename, CropTop, CropLeft, CropRight, CropBottom, ...)
を読み込み、現在開いているカタログ内の写真にファイル名でマッチングして
非破壊でクロップ(Develop設定)を適用する。

元のJPGファイル自体は一切変更しない(Lightroomのカタログ内の編集情報のみ更新)。
「カタログ設定 > メタデータ > XMPへ自動的に変更を書き込む」がオフになっていれば
JPGファイルへの書き戻しも行われない。
--]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'
local LrLogger = import 'LrLogger'

local log = LrLogger('CropFromCSV')
log:enable('logfile')

-- CSVパーサ(カンマ区切り。ダブルクォートで囲まれたフィールド内のカンマに対応)
-- path列にはWindowsのフルパスが入り、カンマを含む場合はクォートされるため、
-- 単純なカンマ分割では壊れる。
local function splitCSVLine(line)
    line = line:gsub("\r$", "")
    local fields = {}
    local pos = 1
    local len = #line
    while true do
        local field
        if string.sub(line, pos, pos) == '"' then
            -- クォート付きフィールド: 閉じクォートまで読む("" はエスケープされた ")
            local buf = {}
            pos = pos + 1
            while pos <= len do
                local ch = string.sub(line, pos, pos)
                if ch == '"' then
                    if string.sub(line, pos + 1, pos + 1) == '"' then
                        table.insert(buf, '"')
                        pos = pos + 2
                    else
                        pos = pos + 1
                        break
                    end
                else
                    table.insert(buf, ch)
                    pos = pos + 1
                end
            end
            field = table.concat(buf)
            local commaPos = string.find(line, ",", pos, true)
            pos = commaPos and (commaPos + 1) or (len + 1)
            table.insert(fields, field)
            if not commaPos then break end
        else
            local commaPos = string.find(line, ",", pos, true)
            if commaPos then
                table.insert(fields, string.sub(line, pos, commaPos - 1))
                pos = commaPos + 1
            else
                table.insert(fields, string.sub(line, pos))
                break
            end
        end
    end
    return fields
end

-- パス比較用の正規化(Windowsの区切り文字ゆれ・大文字小文字を吸収する)
local function normalizePath(p)
    if not p or p == "" then return nil end
    p = p:gsub("\\", "/")
    return p:lower()
end

local function trim(s)
    if not s then return s end
    return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- crop_calculator.py は UTF-8 BOM付き(encoding="utf-8-sig")でCSVを書き出すため、
-- 先頭行にBOM(EF BB BF)が残っていると1列目("filename")の照合に失敗する。
-- そのため読み込み時に先頭行のBOMを取り除く。
local function stripBOM(s)
    if s and s:sub(1, 3) == "\239\187\191" then
        return s:sub(4)
    end
    return s
end

LrTasks.startAsyncTask(function()
    local csvPaths = LrDialogs.runOpenPanel({
        title = "クロップ情報CSVを選択してください",
        canChooseFiles = true,
        canChooseDirectories = false,
        allowsMultipleSelection = false,
        fileTypes = { "csv" },
    })

    if not csvPaths or not csvPaths[1] then
        return
    end
    local csvPath = csvPaths[1]

    -- CSV読み込み
    local lines = {}
    local file = io.open(csvPath, "r")
    if not file then
        LrDialogs.message("エラー", "CSVファイルを開けませんでした: " .. csvPath, "critical")
        return
    end
    for line in file:lines() do
        table.insert(lines, line)
    end
    file:close()

    if #lines < 2 then
        LrDialogs.message("エラー", "CSVにデータ行がありません。", "critical")
        return
    end

    lines[1] = stripBOM(lines[1])
    local header = splitCSVLine(lines[1])
    local colIndex = {}
    for i, h in ipairs(header) do
        colIndex[trim(h)] = i
    end

    local requiredCols = { "filename", "CropTop", "CropLeft", "CropRight", "CropBottom" }
    for _, c in ipairs(requiredCols) do
        if not colIndex[c] then
            LrDialogs.message("エラー", "CSVに必要な列がありません: " .. c, "critical")
            return
        end
    end

    -- カタログ内の写真をインデックス化する。
    -- ファイル名だけの照合は、撮影日フォルダをまたいで同名ファイル(Canonの連番など)が
    -- 存在する場合に別の写真へクロップを適用してしまうため、フルパス照合を優先する。
    local catalog = LrApplication.activeCatalog()
    local allPhotos = catalog:getAllPhotos()
    local photoByPath = {}
    local photoByName = {}
    local nameCount = {}
    for _, photo in ipairs(allPhotos) do
        local fname = photo:getFormattedMetadata('fileName')
        if fname then
            nameCount[fname] = (nameCount[fname] or 0) + 1
            photoByName[fname] = photo
        end
        local ppath = normalizePath(photo:getRawMetadata('path'))
        if ppath then
            photoByPath[ppath] = photo
        end
    end

    -- 適用対象を先に集計(進捗表示・エラー集計用)
    local jobs = {}
    local missingList = {}
    local skippedList = {}
    local ambiguousList = {}   -- 同名ファイルが複数あり、パスでも特定できなかったもの
    local matchedByPath, matchedByName = 0, 0
    local hasPathColumn = colIndex["path"] ~= nil

    for i = 2, #lines do
        local line = lines[i]
        if line and trim(line) ~= "" then
            local fields = splitCSVLine(line)
            local filename = trim(fields[colIndex["filename"]])
            local status = colIndex["status"] and trim(fields[colIndex["status"]]) or "OK"

            if status ~= "OK" then
                table.insert(skippedList, filename)
            else
                local cropTop = tonumber(fields[colIndex["CropTop"]])
                local cropLeft = tonumber(fields[colIndex["CropLeft"]])
                local cropRight = tonumber(fields[colIndex["CropRight"]])
                local cropBottom = tonumber(fields[colIndex["CropBottom"]])

                if cropTop and cropLeft and cropRight and cropBottom then
                    -- 1) フルパスで一意に特定できればそれを使う
                    local photo = nil
                    if hasPathColumn then
                        local csvPathValue = normalizePath(trim(fields[colIndex["path"]] or ""))
                        if csvPathValue then
                            photo = photoByPath[csvPathValue]
                            if photo then matchedByPath = matchedByPath + 1 end
                        end
                    end
                    -- 2) パスで特定できない場合のみファイル名にフォールバックする。
                    --    ただし同名ファイルが複数ある場合は、誤った写真に適用する危険が
                    --    あるためスキップして報告する。
                    if not photo then
                        if (nameCount[filename] or 0) > 1 then
                            table.insert(ambiguousList, filename)
                            log:warn("Ambiguous filename (multiple photos in catalog), skipped: " .. filename)
                        else
                            photo = photoByName[filename]
                            if photo then matchedByName = matchedByName + 1 end
                        end
                    end

                    if photo then
                        table.insert(jobs, {
                            photo = photo,
                            filename = filename,
                            CropTop = cropTop,
                            CropLeft = cropLeft,
                            CropRight = cropRight,
                            CropBottom = cropBottom,
                        })
                    elseif (nameCount[filename] or 0) <= 1 then
                        table.insert(missingList, filename)
                    end
                else
                    table.insert(skippedList, filename)
                end
            end
        end
    end

    if #jobs == 0 then
        LrDialogs.message("適用対象なし",
            string.format("カタログ内でマッチする写真が見つかりませんでした。\n未検出: %d件\n同名ファイルが複数あり特定できず: %d件",
                #missingList, #ambiguousList))
        return
    end

    local pathNote
    if hasPathColumn then
        pathNote = string.format("フルパスで特定: %d件 / ファイル名で特定: %d件", matchedByPath, matchedByName)
    else
        pathNote = "CSVにpath列がないため、ファイル名のみで照合しています。\n"
            .. "(crop_calculator.pyを再実行してCSVを作り直すと、フルパスで正確に特定できます)"
    end

    local confirm = LrDialogs.confirm(
        "クロップを適用しますか?",
        string.format(
            "%d件の写真にクロップを適用します。\n%s\n\n"
            .. "マッチしなかった写真: %d件\n"
            .. "同名ファイルが複数あり特定できずスキップ: %d件\n"
            .. "スキップされた行(検出失敗等): %d件",
            #jobs, pathNote, #missingList, #ambiguousList, #skippedList),
        "適用する", "キャンセル"
    )
    if confirm ~= "ok" then
        return
    end

    local applied, errors, notPersisted = 0, 0, 0

    LrTasks.pcall(function()
        catalog:withWriteAccessDo("CSVからクロップを適用", function()
            for _, job in ipairs(jobs) do
                local ok, err = LrTasks.pcall(function()
                    -- クロップ矩形とCropAngleのみを渡す。
                    -- (CropConstrainAspectRatio等は写真ごとの既存値を尊重し、上書きしない)
                    job.photo:applyDevelopSettings({
                        CropTop = job.CropTop,
                        CropLeft = job.CropLeft,
                        CropRight = job.CropRight,
                        CropBottom = job.CropBottom,
                        CropAngle = 0,
                    })
                end)
                if ok then
                    applied = applied + 1
                    -- 念のため書き込み直後に読み戻して確認する。
                    -- (適用先の写真自体が誤っていると、この検証は一致してしまい
                    --  問題を検出できないため、あくまで補助的なチェック)
                    local readOk, current = LrTasks.pcall(function()
                        return job.photo:getDevelopSettings()
                    end)
                    if readOk and current then
                        local gotLeft = current.CropLeft or 0
                        local gotTop = current.CropTop or 0
                        local gotRight = current.CropRight or 1
                        local gotBottom = current.CropBottom or 1
                        local mismatch = math.abs(gotLeft - job.CropLeft) > 0.001
                            or math.abs(gotTop - job.CropTop) > 0.001
                            or math.abs(gotRight - job.CropRight) > 0.001
                            or math.abs(gotBottom - job.CropBottom) > 0.001
                        if mismatch then
                            notPersisted = notPersisted + 1
                            log:warnf(
                                "%s: applyDevelopSettings did not persist (immediate readback). " ..
                                "wanted L=%.4f T=%.4f R=%.4f B=%.4f / got L=%.4f T=%.4f R=%.4f B=%.4f",
                                job.filename, job.CropLeft, job.CropTop, job.CropRight, job.CropBottom,
                                gotLeft, gotTop, gotRight, gotBottom
                            )
                        end
                    else
                        log:warn(job.filename .. ": getDevelopSettings readback failed: " .. tostring(current))
                    end
                else
                    errors = errors + 1
                    log:error("Failed to apply crop for " .. job.filename .. ": " .. tostring(err))
                end
            end
        end)
    end)

    local msg = string.format(
        "適用完了: %d件(フルパス特定: %d件 / ファイル名特定: %d件)\n" ..
        "  うち反映が確認できなかったもの: %d件(ログファイル参照)\n" ..
        "エラー: %d件\nカタログ内で未検出: %d件\n" ..
        "同名ファイルが複数あり特定できずスキップ: %d件\nスキップ(検出失敗等): %d件",
        applied, matchedByPath, matchedByName, notPersisted, errors,
        #missingList, #ambiguousList, #skippedList
    )
    LrDialogs.message("クロップ適用結果", msg)
end)
