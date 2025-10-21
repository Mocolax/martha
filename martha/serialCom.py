import time
import serial
import struct


class Serialhandler():
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM1', 115200, timeout=0.0025)
        self.last_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        
    def serialSend(self, x, y, z):
            message = struct.pack(
                    'dddddd',
                    x/3,        # x
                    -y/3,       # y
                    -z,         # z
                    False,      # start_button
                    False,      # back_button
                    9912399
                )	
            self.ser.write(message)
            time.sleep(0.01)

    def serialRead(self):
            s = self.ser.read(112)        # read up to ten bytes (timeout)
            if len(s) == 112:
                px = struct.unpack('d',s[0:8])[0]
                py = struct.unpack('d',s[8:16])[0]
                rz = struct.unpack('d',s[16:24])[0]

                vx_read = struct.unpack('d',s[24:32])[0]
                vy_read = struct.unpack('d',s[32:40])[0]
                wz_read = struct.unpack('d',s[40:48])[0]

                velWheel1 = struct.unpack('d',s[48:56])[0]
                velWheel2 = struct.unpack('d',s[56:64])[0]
                velWheel3 = struct.unpack('d',s[64:72])[0]
                velWheel4 = struct.unpack('d',s[72:80])[0]

                control1 = struct.unpack('d',s[80:88])[0]
                control2 = struct.unpack('d',s[88:96])[0]
                control3 = struct.unpack('d',s[96:104])[0]
                control4 = struct.unpack('d',s[104:112])[0]
                
                self.last_values = (px, py, rz, vx_read, vy_read, wz_read)

            return self.last_values

    
    def serialClose(self):
        message = struct.pack(
				'dddddd',
				0.0,      # x
				0.0,     # y
				0.0,    # z
				False,      # start_button
				False,       # back_button
				9912399
			)
        self.ser.write(message)
        self.ser.close()