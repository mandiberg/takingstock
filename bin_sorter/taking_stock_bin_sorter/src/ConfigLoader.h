#pragma once

#include <string>
#include <vector>
#include "BinSorter.h"

struct BinSorterConfig {
    int boxWidth = 1920;
    int boxHeight = 1080;
    std::string videoAssetPath = "videos";
    std::string arrangementsPath = "arrangements";
    std::vector<SizeRatio> sizeRatios;
    int gapFilterThreshold = 1000;   // reject layouts where largest empty rect >= this (px²); 0 = only perfect fill
    int packingStopArea = 1000;     // stop placing when largest placeable item would be < this (px²); prevents infinite tiny items
    int nestingLayers = 1;
    int nestedMinSpaceThreshold = 0;
    float mainBinFillChance = 0.05f;
    float itemBreakScale = 0.45f;
    float itemBreakChance = 0.95f;
    int breakBoxMinItems = 1;
    int breakBoxMaxItems = 4;
    int breakBoxFillAttempts = 5;
    float breakBoxCoverageThreshold = 0.99f;
    int layoutMaxAttempts = 50000;      // max sort() calls per phase before giving up
    int layoutStaleThreshold = 1500;   // stop phase after this many consecutive duplicates
    int layoutPhases = 5;               // number of reseeded phases to explore different regions
    float placementAreaExponent = 1.2f;  // score = area^exp * weight; >1 favors larger items
    int placementTopK = 3;              // randomly pick from top K candidates for variation (1=always best)
};

class ConfigLoader {
public:
    static bool load(const std::string& path, BinSorterConfig& out);
private:
    static std::string trim(const std::string& s);
    static void parseLine(const std::string& line, BinSorterConfig& config);
};
