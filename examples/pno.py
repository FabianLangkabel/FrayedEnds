import madpy

# initialize the PNO interface
madpno = madpy.MadPNO("he 0.0 0.0 0.0")
# compute integrals --> should be deprecated in favor of general Integrator class
c, h, g = madpno.get_integrals()