#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Call MATLAB to print "Hello, World!"
eng.eval("disp('MATLAB Engine for Python successfully started')", nargout=0)

# Stop MATLAB engine
eng.quit()

