@echo off
REM Triple Training Workflow for Windows
REM This shows the complete workflow for training and exporting 3 temperature banks

echo ==============================================
echo Triple Training Workflow for Thermal DPD
echo ==============================================
echo.

REM Step 1: Create models directory
if not exist models mkdir models

REM Step 2: Train each temperature separately
echo Step 1: Training cold temperature network...
python train.py ^
    --temp cold ^
    --output models/dpd_cold.pt ^
    --epochs 200 ^
    --batch-size 128 ^
    --device cuda

echo.
echo Step 2: Training normal temperature network...
python train.py ^
    --temp normal ^
    --output models/dpd_normal.pt ^
    --epochs 200 ^
    --batch-size 128 ^
    --device cuda

echo.
echo Step 3: Training hot temperature network...
python train.py ^
    --temp hot ^
    --output models/dpd_hot.pt ^
    --epochs 200 ^
    --batch-size 128 ^
    --device cuda

REM Step 3: Export all 3 networks
echo.
echo Step 4: Exporting all 3 networks to FPGA weight files...
python export.py ^
    --triple-trained ^
    --checkpoint-cold models/dpd_cold.pt ^
    --checkpoint-normal models/dpd_normal.pt ^
    --checkpoint-hot models/dpd_hot.pt ^
    --output rtl/weights ^
    --format hex bin

echo.
echo ==============================================
echo Triple Training Complete!
echo ==============================================
echo Weight files generated:
echo   - rtl/weights/weights_bank0_cold.hex
echo   - rtl/weights/weights_bank1_normal.hex
echo   - rtl/weights/weights_bank2_hot.hex
echo.
echo Total BRAM: 9.3 KB (3 banks x 1,554 params x 16-bit)
echo.
pause
