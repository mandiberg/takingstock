#include "ConfigLoader.h"
#include "BinSorter.h"
#include "ofMain.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

std::string ConfigLoader::trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void ConfigLoader::parseLine(const std::string& line, BinSorterConfig& config) {
    size_t eq = line.find('=');
    if (eq == std::string::npos) return;
    std::string key = trim(line.substr(0, eq));
    std::string value = trim(line.substr(eq + 1));
    size_t hashPos = value.find('#');
    if (hashPos != std::string::npos)
        value = trim(value.substr(0, hashPos));
    if (key.empty()) return;

    if (key == "BOX_WIDTH") { config.boxWidth = std::stoi(value); return; }
    if (key == "BOX_HEIGHT") { config.boxHeight = std::stoi(value); return; }
    if (key == "VIDEO_ASSET_PATH") { config.videoAssetPath = value; return; }
    if (key == "ARRANGEMENTS_PATH") { config.arrangementsPath = value; return; }
    if (key == "VIDEO_LOOP") {
        std::string v = value;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        config.videoLoop = (v == "1" || v == "true" || v == "yes");
        return;
    }
    if (key == "TRANSITION_TYPE") {
        std::string v = value;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (v == "fade") config.transitionType = TransitionType::Fade;
        else if (v == "jumpcut_to_black") config.transitionType = TransitionType::JumpcutToBlack;
        else config.transitionType = TransitionType::Jumpcut;
        return;
    }
    if (key == "TRANSITION_DURATION_FADE") { config.transitionDurationFade = std::stof(value); return; }
    if (key == "TRANSITION_DURATION_JUMP_TO_BLACK") { config.transitionDurationJumpToBlack = std::stof(value); return; }
    if (key == "TRANSITION_TIMER_MIN") { config.transitionTimerMin = std::stof(value); return; }
    if (key == "TRANSITION_TIMER_MAX") { config.transitionTimerMax = std::stof(value); return; }
    if (key == "MIN_SPACE_THRESHOLD") {
        int v = std::stoi(value);
        config.gapFilterThreshold = v;
        config.packingStopArea = v;
        return;
    }
    if (key == "GAP_FILTER_THRESHOLD") { config.gapFilterThreshold = std::stoi(value); return; }
    if (key == "PACKING_STOP_AREA") { config.packingStopArea = std::stoi(value); return; }
    if (key == "NESTING_LAYERS") { config.nestingLayers = std::stoi(value); return; }
    if (key == "NESTED_MIN_SPACE_THRESHOLD") { config.nestedMinSpaceThreshold = std::stoi(value); return; }
    if (key == "MAIN_BIN_FILL_CHANCE") { config.mainBinFillChance = std::stof(value); return; }
    if (key == "ITEM_BREAK_SCALE") { config.itemBreakScale = std::stof(value); return; }
    if (key == "ITEM_BREAK_CHANCE") { config.itemBreakChance = std::stof(value); return; }
    if (key == "BREAK_BOX_MIN_ITEMS") { config.breakBoxMinItems = std::stoi(value); return; }
    if (key == "BREAK_BOX_MAX_ITEMS") { config.breakBoxMaxItems = std::stoi(value); return; }
    if (key == "BREAK_BOX_FILL_ATTEMPTS") { config.breakBoxFillAttempts = std::stoi(value); return; }
    if (key == "BREAK_BOX_COVERAGE_THRESHOLD") { config.breakBoxCoverageThreshold = std::stof(value); return; }
    if (key == "LAYOUT_MAX_ATTEMPTS") { config.layoutMaxAttempts = std::stoi(value); return; }
    if (key == "LAYOUT_STALE_THRESHOLD") { config.layoutStaleThreshold = std::stoi(value); return; }
    if (key == "LAYOUT_PHASES") { config.layoutPhases = std::stoi(value); return; }
    if (key == "PLACEMENT_AREA_EXPONENT") { config.placementAreaExponent = std::stof(value); return; }
    if (key == "PLACEMENT_TOP_K") { config.placementTopK = std::stoi(value); return; }

    if (key == "SIZE_RATIO") {
        std::istringstream iss(value);
        int w, h;
        float weight, exX, exY;
        if (iss >> w >> h >> weight >> exX >> exY)
            config.sizeRatios.push_back(SizeRatio(w, h, weight, exX, exY));
        return;
    }
}

bool ConfigLoader::load(const std::string& path, BinSorterConfig& out) {
    std::string fullPath = ofToDataPath(path, true);
    std::ifstream f(fullPath);
    if (!f.is_open()) {
        ofLogError("ConfigLoader") << "Cannot open config: " << fullPath;
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        parseLine(line, out);
    }
    return true;
}
