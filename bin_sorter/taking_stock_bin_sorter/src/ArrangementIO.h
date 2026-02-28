#pragma once

#include "BinSorter.h"
#include <string>
#include <vector>

namespace ArrangementIO {
    std::string getArrangementPath(const std::string& arrangementsPath, int boxWidth, int boxHeight, int nestingLayers, int numArrangements);
    std::string findArrangementPath(const std::string& arrangementsPath, int boxWidth, int boxHeight, int nestingLayers);
    bool isValidArrangement(const Arrangement& arr, int boxWidth, int boxHeight);
    bool load(const std::string& path, std::vector<Arrangement>& out);
    bool save(const std::string& path, const std::vector<Arrangement>& arrangements);
}
