#include "VideoAssetPool.h"
#include "ofMain.h"
#include "ofFileUtils.h"
#include <algorithm>

bool VideoAssetPool::isVideoExtension(const std::string& path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower.find(".mp4") != std::string::npos ||
           lower.find(".mov") != std::string::npos ||
           lower.find(".avi") != std::string::npos ||
           lower.find(".mkv") != std::string::npos ||
           lower.find(".webm") != std::string::npos;
}

bool VideoAssetPool::load(const std::string& assetRootPath) {
    pathsByRatio.clear();
    std::string root = ofToDataPath(assetRootPath, true);
    ofDirectory dir(root);
    if (!dir.exists()) {
        ofLogWarning("VideoAssetPool") << "Asset root does not exist: " << root;
        return false;
    }
    dir.listDir();
    dir.sort();  // consistent ordering
    for (int i = 0; i < dir.size(); ++i) {
        if (!dir.getFile(i).isDirectory()) continue;
        std::string folderName = dir.getName(i);
        std::string folderPath = dir.getPath(i);
        ofDirectory subDir(folderPath);
        subDir.listDir();
        subDir.sort();
        std::vector<std::string> videos;
        for (int j = 0; j < subDir.size(); ++j) {
            std::string p = subDir.getPath(j);
            if (subDir.getFile(j).isFile() && isVideoExtension(p))
                videos.push_back(p);
        }
        if (!videos.empty())
            pathsByRatio[folderName] = videos;
    }
    resetUsed();
    return !pathsByRatio.empty();
}

void VideoAssetPool::resetUsed() {
    availableByRatio = pathsByRatio;
}

std::string VideoAssetPool::getVideoPath(int wr, int hr) {
    std::string key = std::to_string(wr) + "_" + std::to_string(hr);
    auto it = pathsByRatio.find(key);
    if (it == pathsByRatio.end() || it->second.empty()) {
        ofLogWarning("VideoAssetPool") << "No video for ratio " << key
            << " (folder missing or empty)";
        return "";
    }
    std::vector<std::string>& available = availableByRatio[key];
    if (available.empty())
        available = it->second;  // reuse when we've exhausted all - "absolutely have to"
    size_t idx = (size_t)(ofRandom(0.0f, (float)available.size()));
    if (idx >= available.size()) idx = available.size() - 1;
    std::string path = available[idx];
    available[idx] = available.back();
    available.pop_back();
    return path;
}

bool VideoAssetPool::hasVideosFor(int wr, int hr) const {
    std::string key = std::to_string(wr) + "_" + std::to_string(hr);
    auto it = pathsByRatio.find(key);
    return it != pathsByRatio.end() && !it->second.empty();
}
