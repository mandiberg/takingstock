#pragma once

#include <string>
#include <vector>
#include <map>

class VideoAssetPool {
public:
    bool load(const std::string& assetRootPath);
    void resetUsed();  // call when starting a new layout - makes all videos available again
    std::string getVideoPath(int wr, int hr);  // picks from unused; reuses only when none left
    bool hasVideosFor(int wr, int hr) const;
private:
    std::map<std::string, std::vector<std::string>> pathsByRatio;
    std::map<std::string, std::vector<std::string>> availableByRatio;  // shrinks as videos are picked
    static bool isVideoExtension(const std::string& path);
};
