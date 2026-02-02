#pragma once

#ifdef BACKEND_CPU
#include "calculate_local_energy_cpu.h"
#elif defined(BACKEND_GPU)
#include "calculate_local_energy.cuh"
#elif defined(BACKEND_DLC)
#include "calculate_local_energy_dlc.h"
#endif

