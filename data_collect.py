import argparse
from X4M03_Raw_Data_Extractor import main, setup_radar, run_radar
import serial
from types import SimpleNamespace
import os
import time

def main():   
    #FOLDER_NAME = "dataset/human_walking"
    FOLDER_NAME = "dataset/human_limping"
    #FOLDER_NAME = "dataset/human_falling"
    os.makedirs(FOLDER_NAME, exist_ok=True)

    # PER DAY STUFF:
    clip_num = 10
    DATA_FILE_NAME = "11_4_2022_%i"%(clip_num)

    options = SimpleNamespace(
        PORT="COM5",
        baseband=False,
        filename=os.path.join(FOLDER_NAME, "%s.txt"%DATA_FILE_NAME),
        fps=20,
        time=10,
    )
    
    time.sleep(10)

    X4 = serial.Serial(
        options.PORT,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=6)

    fps = options.fps
    #time = options.time
        
    setup_radar(X4, fps, options.baseband)
    run_radar(X4, options.filename, fps, options.time)
        

if __name__ == "__main__":
   main()