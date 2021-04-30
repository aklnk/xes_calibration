# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:18:43 2019

@author: akali
"""

import zmq
import numpy as np

###INIT 0MQ Client SIDE
port = "5489"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
print(socket)
socket.connect("tcp://localhost:%s" % port)

#imVert = 516
#imHor = 1556

energy = 9977.55
i0 = 1256.11
#image = np.zeros((imVert, imHor))

#md = dict(
#        dtype = str(image.dtype),
#        shape = image.shape,
#        energy = energy
#        )
#flags = 0
#socket.send_json(md, flags|zmq.SNDMORE)
#socket.send(image, flags, copy=True, track=False)

socket.send_string("NzmqTest_v06")

socket.send_string("E"+str(energy)+"=I"+str(i0))
socket.send_string("E"+str(energy+1)+"=I"+str(i0*2))

socket.send_string("NzmqTest_v06_cycle1")

socket.send_string("E"+str(energy+5)+"=I"+str(i0*3))
socket.send_string("E"+str(energy+10)+"=I"+str(i0*10))