# Tuple JSON Serialization Fixes

## Problem
The frontend was receiving JSON parsing errors like `"Unexpected token '(', "(0, 0, 3)" is not valid JSON"` because Python tuples were being serialized as strings instead of JSON arrays.

## Root Causes Found and Fixed

### 1. LND Routes - convert_to_json_safe functions
**File**: `app/routes/lnd.py`
**Issue**: Multiple `convert_to_json_safe` functions lacked tuple handling
**Fix**: Added `elif isinstance(obj, tuple): return list(obj)` to all convert_to_json_safe functions

### 2. LND Service - facility location returns
**File**: `app/services/lnd_service.py`
**Issue**: Direct tuple returns in facility location calculations
**Fixes**:
- Line 867: `'facility_location': list(facility_location)` instead of tuple
- Line 374: `facility_locations = [[float(X[j]), float(Y[j])] for j in range(num_facilities)]` instead of tuples
- Lines with `facility_locations.append()` now append lists instead of tuples

### 3. Global Utility Function
**File**: `app/utils.py` (created)
**Purpose**: Centralized tuple-to-list conversion for JSON serialization
**Function**: `convert_to_json_safe(obj)` handles all problematic types including tuples

### 4. Analytics Routes Enhancement
**File**: `app/routes/analytics.py`
**Enhancement**: Added import of utility function and applied it to chart data returns

## Types of Data Fixed
- Coordinate tuples: `(lat, lon)` → `[lat, lon]`
- Facility locations: `[(x1, y1), (x2, y2)]` → `[[x1, y1], [x2, y2]]`
- Chart coordinates: `(0, 0, 3)` → `[0, 0, 3]`
- Nested structures containing tuples

## Testing
Created `test_tuple_fix.py` which confirms:
- Basic tuples convert correctly
- Nested structures with tuples convert correctly  
- Lists containing tuples convert correctly
- All converted data can be JSON serialized successfully

## API Endpoints Most Likely Affected
- LND facility location optimization endpoints
- Analytics chart generation endpoints
- Any endpoint returning coordinate or geometric data

## Result
All Python tuples should now be properly converted to JSON arrays before being sent to the frontend, eliminating the JSON parsing errors.