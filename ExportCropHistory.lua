--[[
ExportCropHistory.lua

汎用ポートレートクロップモデル(train_general_crop_model.py)の学習データを
作るためのプラグイン。カタログ内の写真から、既に手動でクロップ済みの
Developクロップ値(crs:CropTop/Left/Right/Bottom, CropAngle)と、
元ファイルパスをCSVに書き出す。

対象は「選択中の写真」(未選択の場合はカタログの現在のソース内の全写真、
LrCatalog:getTargetPhotos() の挙動に準拠)。書き出す前にフォルダ/コレクションを
選んでおくことで、対象を絞り込める。

出力CSV列:
    filename, path, raw_W, raw_H, CropTop, CropLeft, CropRight, CropBottom,
    CropAngle, is_cropped

is_cropped は「デフォルト(CropLeft=0,Top=0,Right=1,Bottom=1)からクロップ変更
されているか」を表す(0/1)。train_general_crop_model.py 側で is_cropped=1 かつ
CropAngle がほぼ0の行のみを学習に使う。

このプラグイン自体はファイルの読み書きを一切行わない(カタログの
メタデータを読み取るのみ)。
--]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'
local LrLogger = import 'LrLogger'

local log = LrLogger('ExportCropHistory')
log:enable('logfile')

local EPS = 1e-6

local function isDefaultCrop(l, t, r, b)
    return math.abs(l - 0) < EPS and math.abs(t - 0) < EPS
        and math.abs(r - 1) < EPS and math.abs(b - 1) < EPS
end

-- CSVフィールドのエスケープ(カンマ・ダブルクォート・改行を含む場合のみクォートする)
local function csvField(v)
    if v == nil then
        return ""
    end
    local s = tostring(v)
    if s:find('[,"\r\n]') then
        s = '"' .. s:gsub('"', '""') .. '"'
    end
    return s
end

LrTasks.startAsyncTask(function()
    local catalog = LrApplication.activeCatalog()
    local targetPhotos = catalog:getTargetPhotos()

    if not targetPhotos or #targetPhotos == 0 then
        LrDialogs.message("対象なし", "書き出し対象の写真がありません。"
            .. "フォルダ/コレクションを選択するか、写真を選択してから実行してください。")
        return
    end

    -- 集計(確認ダイアログ用)
    local rows = {}
    local croppedCount = 0
    local rotatedCount = 0

    for _, photo in ipairs(targetPhotos) do
        local ds = photo:getDevelopSettings()
        local cropLeft = ds.CropLeft or 0
        local cropTop = ds.CropTop or 0
        local cropRight = ds.CropRight or 1
        local cropBottom = ds.CropBottom or 1
        local cropAngle = ds.CropAngle or 0

        local cropped = not isDefaultCrop(cropLeft, cropTop, cropRight, cropBottom)
        if cropped then
            croppedCount = croppedCount + 1
            if math.abs(cropAngle) > 0.01 then
                rotatedCount = rotatedCount + 1
            end
        end

        local filename = photo:getFormattedMetadata('fileName') or ""
        local path = photo:getRawMetadata('path') or ""
        local dims = photo:getRawMetadata('dimensions') or {}

        table.insert(rows, {
            filename = filename,
            path = path,
            raw_W = dims.width,
            raw_H = dims.height,
            CropTop = cropTop, CropLeft = cropLeft,
            CropRight = cropRight, CropBottom = cropBottom,
            CropAngle = cropAngle,
            is_cropped = cropped and 1 or 0,
        })
    end

    local confirm = LrDialogs.confirm(
        "クロップ履歴を書き出しますか?",
        string.format(
            "対象写真: %d件\n手動トリミング済み: %d件\n"
            .. "  (うち回転クロップ: %d件。学習時には除外されます)\n\n"
            .. "保存先のCSVファイルを選択してください。",
            #rows, croppedCount, rotatedCount
        ),
        "CSVを保存...", "キャンセル"
    )
    if confirm ~= "ok" then
        return
    end

    local savePath = LrDialogs.runSavePanel({
        title = "クロップ履歴CSVの保存先",
        canCreateDirectories = true,
        requiredFileType = "csv",
        prompt = "保存",
    })
    if not savePath then
        return
    end

    local file, openErr = io.open(savePath, "w")
    if not file then
        LrDialogs.message("エラー", "CSVファイルを書き込めませんでした: " .. tostring(openErr), "critical")
        return
    end

    file:write("filename,path,raw_W,raw_H,CropTop,CropLeft,CropRight,CropBottom,CropAngle,is_cropped\n")
    for _, r in ipairs(rows) do
        file:write(table.concat({
            csvField(r.filename), csvField(r.path),
            csvField(r.raw_W), csvField(r.raw_H),
            csvField(r.CropTop), csvField(r.CropLeft),
            csvField(r.CropRight), csvField(r.CropBottom),
            csvField(r.CropAngle), csvField(r.is_cropped),
        }, ",") .. "\n")
    end
    file:close()

    LrDialogs.message("書き出し完了",
        string.format("%d件を書き出しました(手動トリミング済み: %d件)。\n保存先: %s",
            #rows, croppedCount, savePath))
end)
