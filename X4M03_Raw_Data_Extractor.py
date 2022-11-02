#Control the X4M03 Radar Using Serial Communication to Extract Raw Data
#Written By Abdel-Kareem Moadi, Vijay Rajagopal University of Tennessee Knoxville
import serial
import time
import numpy as np
import struct
import math
from struct import *
from optparse import OptionParser

## Serial Communication Definitions for X4 Chip
XT_START  = b'\x7D'
XT_STOP   = b'\x7E'
XT_ESCAPE = b'\x7F'

XTS_FLAGSEQUENCE_START_NOESCAPE = b'\x7C\x7C\x7C\x7C'

XTS_SPC_X4DRIVER = b'\x50'
XTS_SPCX_SET     = b'\x10'
XTS_SPCX_GET     = b'\x11'

XTS_SPC_MOD_RESET = b'\x22'

XTS_SPCXI_FPS             = b'\x00\x00\x00\x10'
XTS_SPCXI_PULSEPERSTEP    = b'\x00\x00\x00\x11'
XTS_SPCXI_ITERATIONS      = b'\x00\x00\x00\x12'
XTS_SPCXI_DOWNCONVERSION  = b'\x00\x00\x00\x13'
XTS_SPCXI_FRAMEAREA       = b'\x00\x00\x00\x14'
XTS_SPCXI_DACSTEP         = b'\x00\x00\x00\x15'
XTS_SPCXI_DACMIN          = b'\x00\x00\x00\x16'
XTS_SPCXI_DACMAX          = b'\x00\x00\x00\x17'
XTS_SPCXI_FRAMEAREAOFFSET = b'\x00\x00\x00\x18'
XTS_SPCXI_PRFDIV          = b'\x00\x00\x00\x25'

XTS_SPR_APPDATA = b'\x50'
XTS_SPR_SYSTEM  = b'\x30'

XTS_SPR_ACK   = b'\x10'
XTS_SPR_ERROR = b'\x20'

XTS_SPRS_BOOTING = b'\x00\x00\x00\x10'
XTS_SPRS_READY   = b'\x00\x00\x00\x11'

XTS_RESPONSE = b'\x7D\x10\x6D\x7E'

booting = XT_START + XTS_SPR_SYSTEM + bytes(reversed(XTS_SPRS_BOOTING)) + b'\x5D' + XT_STOP #Compute the booting successful message
ready = XT_START + XTS_SPR_SYSTEM + bytes(reversed(XTS_SPRS_READY)) + b'\x5C' + XT_STOP #Compute the radar ready message

def get_crc(X4,msg): #Calculates the CRC byte for the currennt command
    crc = 0
    for b in range(len(msg)):
        crc ^= msg[b]
        
    return crc.to_bytes(1,byteorder='big')
    
def send_command(X4,msg, canFail=False): #Sends the command to the radar and waits for a response
    X4.write(msg)
    response = X4.read(4)
    
    if response != XTS_RESPONSE:
        if canFail:
            print("WARNING: XTS_RESPONSE not received. CanFail set to true so ignoring...")
        else:
            X4.close()
            #quit()
            raise Exception("Error, Closing Radar")


def set_fps(X4,fps, canFail=False): #Set the fps for the radar
    val = bytes(struct.pack("<f",fps))
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_FPS)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg, canFail=canFail)
    
def set_pulses_per_step(X4,pps): #Set the pulses per step for the radar
    val = pps.to_bytes(4,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_PULSEPERSTEP)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_iterations(X4,iterations): #Set the iterations for the radar
    val = iterations.to_bytes(4,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_ITERATIONS)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_downconversion(X4,baseband): #Decide whether we should downconvert the radar data
    val = baseband.to_bytes(1,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_DOWNCONVERSION)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_dac_step(X4,dac_step): #Set the analog to digital converter step size
    val = dac_step.to_bytes(4,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_DACSTEP)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_dac_min(X4,dac_min): #Set the minimum value that will register for the analog to digital converter
    val = dac_min.to_bytes(4,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_DACMIN)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_dac_max(X4,dac_max): #Set the maximum value that will register for the analog to digital converter
    val = dac_max.to_bytes(4,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_DACMAX)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_frame_offset(X4,offset): #Set the frame offset for the radar, usually 0.18 m
    val = bytes(struct.pack("<f",offset))
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_FRAMEAREAOFFSET)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_frame_area(X4,frame_start, frame_end): #Set the frame area for the radar
    val = bytes(struct.pack("<f",frame_start))
    val2 = bytes(struct.pack("<f",frame_end))
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_FRAMEAREA)) + val + val2
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def set_prf_div(X4,div): #Set the pulse repitition frequency for the radar
    val = div.to_bytes(4,byteorder='little')
    msg = XT_START + XTS_SPC_X4DRIVER + XTS_SPCX_SET + bytes(reversed(XTS_SPCXI_PRFDIV)) + val
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
def reset_radar(X4): #Reset the radar, reccomended before changing radar settings
    msg = XT_START + XTS_SPC_MOD_RESET
    
    crc = get_crc(X4,msg)
    
    msg = msg + crc + XT_STOP
    
    send_command(X4,msg)
    
    X4.close()
    
    time.sleep(2)
    
    X4.open()
    
    response = X4.read(8)
    if response == booting:
        print("Booting...")
    else:
        print("Error, Could not Boot!")
        X4.close()
        quit()
        
    response = X4.read(8)
    if response == ready:
        print("Radar Ready!")
    else:
        print("Error After Booting!")
        X4.close()
        quit()
        
def setup_radar(X4,fps, baseband): #Sets the radar parameters   
    #PRF_DIV = 16 #How much do you want to divide the PRF
    PRF_DIV = 4
    PRF = 143e6/PRF_DIV #Calculate the PRF
    iterations = 64
    dac_max = 1100
    dac_min = 949
    duty_cycle = 0.95 #Reccomended to remain at 95% duty cycle
    
    PPS = math.floor((PRF/(iterations*fps*((dac_max-dac_min)+1)))*duty_cycle) #Calculate Pule Per Second depending on FPS inputted
    print("pps", PPS)

    #Check if PPS is valid
    if PPS > 65535:
        print("Error, FPS too low")
        X4.close()
        quit()
    elif PPS < 1:
        print("Error, FPS too high")
        X4.close()
        quit()
    
    #Use the written functions to set the radar parameters
    reset_radar(X4)
    set_pulses_per_step(X4,PPS)
    set_downconversion(X4,baseband)
    set_iterations(X4,iterations)
    set_dac_min(X4,dac_min)
    set_dac_max(X4,dac_max)
    set_dac_step(X4,1)
    set_prf_div(X4,PRF_DIV)
    set_frame_offset(X4,0.18)
    set_frame_area(X4,0.0, 9.8)
        
def run_radar(X4, filename, fps, time): #Start the radar data collection and save to text file
    set_fps(X4,fps)
    
    X4.flush()
    
    data = X4.read(4)
    print("No Escape")
    print(data.hex())

    data = X4.read(4)
    print("Packet Length:")
    packet_length = int.from_bytes(data, byteorder='little')
    print(packet_length)

    data = X4.read(1)
    print("Reserved")
    print(data.hex())

    data = X4.read(1)
    print("XTS_SPR_DATA:")
    print(data.hex())

    data = X4.read(1)
    print("XTS_SPRD_FLOAT")
    print(data.hex())

    data = X4.read(4)
    print("Content ID:")
    print(data.hex())

    data = X4.read(4)
    print("Frame:")
    frame = int.from_bytes(data, byteorder='little')
    print(frame)

    data = X4.read(4)
    print("Length:")
    data_length = int.from_bytes(data, byteorder='little')
    print(data_length)
    
    Raw_Data = np.empty([data_length, time*fps],dtype=float)
    
    for j in range(time*fps):
        if j != 0:
            Packet_Header = X4.read(23)
        for i in range(data_length):
            if i == 0:
                Raw_Data[i,j], = unpack("<f",X4.read(4))
            else:
                Raw_Data[i,j], = unpack("<f",X4.read(4))
            
    print(Raw_Data.shape)
    np.savetxt(filename,Raw_Data) 

    set_fps(X4,0, canFail=True)
    X4.close()

def main():   
    parser = OptionParser()
    parser.add_option(
        "-p",
        "--port",
        dest="PORT",
        help="COM Port of Radar",
        metavar="PORT")
    parser.add_option(
        "-b",
        "--baseband",
        action="store_true",
        default=False,
        dest="baseband",
        help="Enable baseband, RF is default")
    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        metavar="FILE",
        help="File where the recording will be saved")
    parser.add_option(
        "-r",
        "--record",
        default=20,
        dest="fps",
        help="Set recording fps. Default is 20")
    parser.add_option(
        "-t",
        "--time",
        default=1,
        dest="time",
        help="Set the time you want to record in seconds. Default is 1")
        
        
    (options, args) = parser.parse_args()
    if not options.PORT:
        parser.error("Missing -d. See --help.")
    if not options.filename:
        parser.error("Missing -f. See --help.")
    if options.fps:
        fps = int(options.fps)
    if options.time:
        time = int(options.time)

    X4 = serial.Serial(
        options.PORT,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=2)
        
    setup_radar(X4, fps, options.baseband)
    run_radar(X4, options.filename, fps, time)
        

if __name__ == "__main__":
   main()