import serial

s = serial.Serial('/dev/tty.usbmodem1101')
c = 0
i = 0

def send(st):
    print(st)
    s.write(bytes(st + '\n\r', 'utf-8'))

cmd = ''
pix = []

while True:
    if c == 0:
        c = 1
    else:
        c = 0

    for x in range(64):
        for y in range(32):
            i += 1
            st = f"{x} {y} {c}"
            pix.append(st)

            if i % 1000 == 0:
                cmd = '$'.join(pix)
                send(cmd)
                cmd = ''
