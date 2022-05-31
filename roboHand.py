import time
from tqdm import tqdm
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

class Robohand:
    def __init__(self):
        self.results = {
            "Sequence_number" :[],
            "AcX"  :[],
            "AcY" : [],
            "AcZ" : [],
            "GyX" : [],
            "GyY" : [],
            "GyZ" : []
        }
        self.ser = None

    def find_port(self):
        for i in range(0,21):
            try:
                ser = serial.Serial('COM' + str(i), 57600)
                testline = ser.readline()
                break
            except:
                # tento port to není
                pass
        else:
            print("Port not found")
            raise IOError
        self.ser = ser

    def read_data(self):
        result = {}
        serial_line = self.ser.readline().decode("utf-8")
        try:
            first_char = serial_line[0]
            if first_char == ">":
                serial_line = serial_line.replace(">", "")
                # mpu status
                serial_line = serial_line[1:-2]
                data = serial_line.split(",")
                data = [int(i, 16) for i in data]


                self.results["Sequence_number"].append(data[0])
                self.results["AcX"].append(data[1])
                self.results["AcY"].append(data[2])
                self.results["AcZ"].append(data[3])
                self.results["GyX"].append(data[4])
                self.results["GyY"].append(data[5])
                self.results["GyZ"].append(data[6])
                """
                if data[7] != (sum(data[0:6]) % 65535):
                    print("Bad checksum")
                    continue
                """
            if first_char == "!":
                # state change
                print("Command not implemented")
        except Exception:
            print("Malformed command")
            pass

    def set_figure(self):
        style.use('fivethirtyeight')

        fig = plt.figure()
        self.ax1 = fig.add_subplot(1, 1, 1)

    def run(self):
        for i in tqdm(range(100000)):
            self.read_data()

        plt.plot(self.results["AcX"])
        #plt.plot(self.results["AcY"])
        #plt.plot(self.results["AcZ"])
        plt.show()
        arr = np.array(self.results["AcX"])
        np.save("bad_move_2",arr)
if __name__ == '__main__':
    obj = Robohand()
    obj.find_port()
    obj.run()