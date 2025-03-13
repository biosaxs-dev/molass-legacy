# coding: utf-8
"""

    ファイル名：   MachineTypes.py

    処理内容：

        マシンタイプ（Chassis Type）などの識別

    Copyright (c) 2015-2019, Masatsuyo Takahashi, KEK-PF

"""

import wmi

CHASSIS_TYPE_NAMES = {
    1   :   "Other",
    2   :   "Unknown",
    3   :   "Desktop",
    4   :   "Low Profile Desktop",
    5   :   "Pizza Box",
    6   :   "Mini Tower",
    7   :   "Tower",
    8   :   "Portable",
    9   :   "Laptop",
    10  :   "Notebook",
    11  :   "Hand Held",
    12  :   "Docking Station",
    13  :   "All in One",
    14  :   "Sub Notebook",
    15  :   "Space-Saving",
    16  :   "Lunch Box",
    17  :   "Main System Chassis",
    18  :   "Expansion Chassis",
    19  :   "SubChassis",
    20  :   "Bus Expansion Chassis",
    21  :   "Peripheral Chassis",
    22  :   "Storage Chassis",
    23  :   "Rack Mount Chassis",
    24  :   "Sealed-Case PC",
    }

def get_chassistype_name():
    wmi_ = wmi.WMI()
    type = None
    for enclosure in wmi_.Win32_SystemEnclosure():
        type = enclosure.ChassisTypes[0]
        break;
    assert( type != None )
    return CHASSIS_TYPE_NAMES[ type ]

def get_display_resolution():
    wmi_ = wmi.WMI()
    width, height = None, None
    for config in wmi_.Win32_DisplayConfiguration():
        width   = config.PelsWidth
        height  = config.PelsHeight
        # print( width, height )
        break
    return ( width, height )

monitors = []

def get_monitors():
    global monitors

    if len( monitors ) > 0: return monitors

    wmi_ = wmi.WMI()
    # for info in wmi_.Win32_DisplayConfiguration():
    # for info in wmi_.Win32_DisplayControllerConfiguration():
    # for info in wmi_.Win32_VideoController():
    # for info in wmi_.Win32_VideoConfiguration():
    # for info in wmi_.Win32_VideoSettings():
    # for info in wmi_.Win32_DesktopMonitor():
    monitors = []
    for info in wmi_.Win32_PnPEntity():
        if info.Service == 'monitor':
            # TODO: set where condition to the above call.
            # print( info.HardwareID )
            monitors.append( info.HardwareID[0].split( '\\' )[1]  )

    return monitors

def get_Win32_Processor_str():
    wmi_ = wmi.WMI()
    return str(wmi_.Win32_Processor()[0])

def is_on_virtual_machine():
    s = get_Win32_Processor_str()
    return s.find('ProcessorId') < 0

def get_cpuid():
    import re
    s = get_Win32_Processor_str()
    cpuid_re = re.compile(r'ProcessorId = "(\w+)"')
    m = cpuid_re.search(s)
    return m.group(1) if m else None

def get_Win32_ComputerSystem_str():
    wmi_ = wmi.WMI()
    return str(wmi_.Win32_ComputerSystem()[0])

def get_bootup_state():
    import re
    s = get_Win32_ComputerSystem_str()
    # print(s)
    state_re = re.compile(r'BootupState = "(.+)";')
    m = state_re.search(s)
    if m:
        return m.group(1)
