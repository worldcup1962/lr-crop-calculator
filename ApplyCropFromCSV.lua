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

-- 簡易CSVパーサ(カンマ区切り、フィールド内カンマ・改行なし前提)
local function splitCSVLine(line)
    line = line:gsub("\r$", "")
    local fields = {}
    local start = 1
    while true do
        local commaPos = string.find(line, ",", start, true)
        if commaPos then
            table.insert(fields, string.sub(line, start, commaPos - 1))
            start = commaPos + 1
        else
            table.insert(fields, string.sub(line, start))
            break
        end
    end
    return fields
end

local function trim(s)
    if not s then return s end
    return (s:gsub("^%s*(.-)%s*$", "%1"))
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

    -- UTF-8 BOM (EF BB BF) が先頭行に付いている場合は除去する
    -- (crop_calculator.py は utf-8-sig で書き出しているため)
    local BOM = "\239\187\191"
    if lines[1] and string.sub(lines[1], 1, 3) == BOM then
        lines[1] = string.sub(lines[1], 4)
    end

    if #lines < 2 then
        LrDialogs.message("エラー", "CSVにデータ行がありません。", "critical")
        return
    end

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

    -- カタログ内の写真をファイル名でインデックス化
    local catalog = LrApplication.activeCatalog()
    local allPhotos = catalog:getAllPhotos()
    local photoByName = {}
    for _, photo in ipairs(allPhotos) do
        local fname = photo:getFormattedMetadata('fileName')
        photoByName[fname] = photo
    end

    -- 適用対象を先に集計(進捗表示・エラー集計用)
    local jobs = {}
    local missingList = {}
    local skippedList = {}

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
                    local photo = photoByName[filename]
                    if photo then
                        table.insert(jobs, {
                            photo = photo,
                            filename = filename,
                            CropTop = cropTop,
                            CropLeft = cropLeft,
                            CropRight = cropRight,
                            CropBottom = cropBottom,
                        })
                    else
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
            string.format("カタログ内でマッチする写真が見つかりませんでした。\n未検出: %d件", #missingList))
        return
    end

    local confirm = LrDialogs.confirm(
        "クロップを適用しますか?",
        string.format("%d件の写真にクロップを適用します。\nマッチしなかった写真: %d件\nスキップされた行(検出失敗等): %d件",
            #jobs, #missingList, #skippedList),
        "適用する", "キャンセル"
    )
    if confirm ~= "ok" then
        return
    end

    local applied, errors = 0, 0

    LrTasks.pcall(function()
        catalog:withWriteAccessDo("CSVからクロップを適用", function()
            for _, job in ipairs(jobs) do
                local ok, err = LrTasks.pcall(function()
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
                else
                    errors = errors + 1
                    log:error("Failed to apply crop for " .. job.filename .. ": " .. tostring(err))
                end
            end
        end)
    end)

    local msg = string.format(
        "適用完了: %d件\nエラー: %d件\nカタログ内で未検出: %d件\nスキップ(検出失敗等): %d件",
        applied, errors, #missingList, #skippedList
    )
    LrDialogs.message("クロップ適用結果", msg)
end)
