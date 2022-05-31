import serial

for i in range(4, 21):
    try:
        ser = serial.Serial('COM' + str(i), 57600)
        testline = ser.readline()
        break
    except:
        #tento port to není
        pass
else:
    print("Port not found")

results = {}

while True:
    serial_line = ser.readline().decode("utf-8")
    try:
        first_char = serial_line[0]
        if first_char == ">":
            serial_line = serial_line.replace(">","")
            #mpu status
            serial_line = serial_line[1:-2]
            data = serial_line.split(",")
            data = [int(i,16) for i in data]

            results["Sequence number"] = data[0]
            results["AcX"] = data[1]
            results["AcY"] = data[2]
            results["AcZ"] = data[3]
            results["GyX"] = data[4]
            results["GyY"] = data[5]
            results["GyZ"] = data[6]
            results["Checksum"] = data[7]
            """
            if data[7] != (sum(data[0:6]) % 65535):
                print("Bad checksum")
                continue
            """
        if first_char == "!":
            # state change
            print("Command not implemented")
            continue
    except Exception:
        print("Malformed command")
        pass