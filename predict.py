import os
import sys
import re
from subprocess import Popen, PIPE, STDOUT

def predict(text="Hello world"):

    #text = "I think as the accepted answer, this should contain at least the same amount of detail as @Harley. This is more of a personal request, but I think the newer docs present the information better. "
    command = ['python', 'PILD.py', text]
    proc = Popen(command, stdout=PIPE)
    response = []
    for line in proc.stdout:
        response.append(line)

    if not response:
      return "Invalid data!"
    else:
      r = response[0]
      number = re.findall(b'(0|1).0', r)
      return number


if __name__ == "__main__":
  print(predict())
