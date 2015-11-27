
mc = {'b' :(77, 117, 179),
      'r' :(210, 88, 88),
      'k' :(38,35,35),
      'white':(255,255,255),
     'grey':(197,198,199)}

for key in mc.keys():
    mc[key] = [x / 255.0 for x in mc[key]]
    print key, mc[key]
