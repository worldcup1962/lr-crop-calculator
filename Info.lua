return {
    LrSdkVersion = 6.0,
    LrSdkMinimumVersion = 6.0,
    LrToolkitIdentifier = 'com.example.cropfromcsv',
    LrPluginName = 'Crop From CSV',
    LrPluginInfoUrl = '',

    LrExportMenuItems = {
        {
            title = "CSVからクロップを適用...",
            file = "ApplyCropFromCSV.lua",
        },
        {
            title = "クロップ履歴をCSVに書き出す...",
            file = "ExportCropHistory.lua",
        },
    },

    VERSION = { major = 2, minor = 0, revision = 0, build = 0 },
}
