import numpy
import numpy as np
mat21=numpy.array([[-0.200239 ,-0.884322 ,0.421757 ,-266.547 ],
[0.435843 ,0.305135 ,0.846719 ,-533.19 ],
[-0.877465,0.353366,0.324326,454.406 ],
[0.0,0.0,0.0,1.0]])
mat22=numpy.array([[0.0333426,-0.891497,0.451799,-295.083 ],
[0.428953,0.421062,0.799191,-501.735 ],
[-0.902711,0.167153,0.396449,407.882 ],
[0.0,0.0,0.0,1.0]])
mat23=numpy.array([[0.264733,-0.847168,0.460677 ,-309.358 ],
[0.392519,0.531021,0.750963,-468.705 ],
[-0.880821,-0.01798,0.473108,356.303 ],
[0.0, 0.0, 0.0, 1.0]])
mat22R=mat22[0:3,0:3]
mat23R=mat23[0:3,0:3]
mat22t=mat22[0:3,3]
mat23t=mat23[0:3,3]
matR = numpy.dot(mat23R,np.transpose(mat22R))
print("相对的旋转矩阵=",end="")
print(matR)
print("相对的归一化平移=",end="")
matt=mat23t-mat22t
matt=matt/((matt[0]*matt[0]+matt[1]*matt[1]+matt[2]*matt[2])**0.5)
print(matt)