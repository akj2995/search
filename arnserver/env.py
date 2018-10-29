__author__ = 'thingscare'

import os
import sys
import platform
from subprocess import Popen, PIPE
import socket
from datetime import datetime
import base64
from uuid import getnode as get_mac
from ConfigParser import ConfigParser
import time
import select
import struct

from tzlocal import get_localzone
from cpuinfo import cpuinfo

current_dir = os.path.dirname(os.path.realpath(__file__))
up_dir = os.path.dirname(current_dir)
sys.path.append(up_dir + '/lib')

try:
    import _winreg
    import psutil
except ImportError:
    pass

ICMP_ECHO_REQUEST = 8 # Seems to be the same on Solaris.


class Env(object):
    def __init__(self):
        pass

    def get_cpu_usage(self):
        return 0

    @staticmethod
    def get_env():
        cur_system = platform.system()

        if cur_system == 'Linux':
            return EnvLinux()
        elif cur_system == 'Darwin':
            return EnvMac()
        elif cur_system == 'Windows':
            return EnvWindows()

    @staticmethod
    def get_cur_system():
        return platform.system()

    def get_hostname(self):
        return socket.gethostname()

    def get_serial_number(self):
        return ""

    def get_hw_address(self):
        return ""

    def find_cpu_temp_hardware(self):
        return 0

    def get_cpu_temp(self, hardware=None):
        return 0

    def get_hdd_temp(self):
        return 0

    def get_fan_speed(self):
        return 0

    def get_product_name(self):
        return ""

    def get_system_info(self):
        return ""

    def get_utc_now(self):
        return str(datetime.utcnow())

    def get_system_now(self):
        return str(datetime.now())

    def get_now_spot(self):
        return "%s" % str(get_localzone())

    def get_timezone(self):
        return str(time.tzname[0])

    def get_utc_standard_time(self):
        local = get_localzone()
        utc_times = str(datetime.now(local))

        plustime = utc_times.split('+')
        if len(plustime) is 2:
            return "UTC%s%s" % ('+',plustime[1])
        else:
            minustime = utc_times.split('-')
            return "UTC%s%s" % ('-',minustime[3])

    def get_uptime(self):
        return ""

    def get_distribution(self):
        return ""

    def get_application_status(self, app_path):
        return ""

    def launch_application(self, app_path):
        return False

    def stop_application(self, app_path):
        return False

    def screen_capture(self, capture_path):
        return False

    def read_screen_capture(self, capture_path):
        return ""

    def get_memory_usage(self):
        return 0.0

    def get_hdd_usage(self):
        return 0.0

FBGRAB = "fbgrab"


class EnvLinux(Env):
    def get_kernel_version(self):
        return os.uname()[2]

    def restart_mediaplayer(self):
        time = 3
        restart = 'shutdown -r -t %d' % time
        os.system(restart)

    def get_physical_total_memory(self):
        mem = psutil.virtual_memory()
        return str(int(mem.total) / (1024*1024))

    def get_physical_total_hdd(self):
        disk = psutil.disk_usage('/')
        return str(int(disk.total) / (1024*1024))

    def get_system_processor(self):
        info = cpuinfo.get_cpu_info()
        return str(info['brand'])

    def get_os_version(self):

        sys_info = dict()

        sys_info['build'] = platform.release()
        sys_info['service_pack'] = platform.uname()[3]

        return sys_info

    def get_cpu_usage(self):
        try:
            process1 = Popen(["top", "-b", "-n", "1"], stdout=PIPE)
            process2 = Popen(["grep", "-i", "cpu(s)"], stdin=process1.stdout, stdout=PIPE)
            process3 = Popen(["awk", "{print $8}"], stdin=process2.stdout, stdout=PIPE)
            cpu_result = Popen(["awk", "{print 100-$1}"], stdin=process3.stdout, stdout=PIPE)

            std_out, std_err = cpu_result.communicate()

            return float(std_out.replace("\n", ""))


        except RuntimeError:
            print "RuntimeError"
            return 0.0
        except ValueError:
            print "ValueError"
            return 0.0

    def get_memory_usage(self):
        """
        $ free
             total       used       free     shared    buffers     cached
Mem:       8957952    5539248    3418704          0     645848    2917956
-/+ buffers/cache:    1975444    6982508
Swap:      9699320          0    9699320
        """
        try:
            mem_result = Popen(
                ["free"],
                shell=False,
                stdout=PIPE)

            std_out, std_err = mem_result.communicate()
            lines = std_out.split("\n")
            line = lines[1].split()
            total = float(line[1])
            used = float(line[2])
            free = float(line[3])

            return used / total * 100.0
        except FloatingPointError:
            return 0.0
        except ZeroDivisionError:
            return 0.0
        except ValueError:
            return 0.0

    def get_hdd_usage(self):
        try:
            hdd_result = Popen(["df", "/"], shell=False, stdout=PIPE)

            std_out, std_err = hdd_result.communicate()
            lines = std_out.split("\n")
            fields = lines[1].split()

            total = float(fields[1])
            used = float(fields[2])

            return used / total * 100
        except IndexError:
            return 0
        except RuntimeError:
            return 0
        except ValueError:
            return 0

    def find_cpu_temp_hardware(self):
        if os.path.exists("/sys/devices/LNXSYSTM:00/LNXTHERM:00/LNXTHERM:01/thermal_zone/temp") == True:
            return  4

        elif os.path.exists("/sys/bus/acpi/devices/LNXTHERM:00/thermal_zone/temp") == True:
            return  5

        elif os.path.exists("/proc/acpi/thermal_zone/THM0/temperature") == True:
            return  1

        elif os.path.exists("/proc/acpi/thermal_zone/THRM/temperature") == True :
            return  2

        elif os.path.exists("/proc/acpi/thermal_zone/THR1/temperature") == True :
            return  3

        else:
            return 0

    def get_cpu_temp(self, hardware=None):
        try:
            if hardware is None:
                hardware = self.find_cpu_temp_hardware()

            if hardware == 1 :
                temp = open("/proc/acpi/thermal_zone/THM0/temperature").read().strip().lstrip('temperature :').rstrip(' C')
            elif hardware == 2 :
                temp = open("/proc/acpi/thermal_zone/THRM/temperature").read().strip().lstrip('temperature :').rstrip(' C')
            elif hardware == 3 :
                temp = open("/proc/acpi/thermal_zone/THR1/temperature").read().strip().lstrip('temperature :').rstrip(' C')
            elif hardware == 4 :
                temp = open("/sys/devices/LNXSYSTM:00/LNXTHERM:00/LNXTHERM:01/thermal_zone/temp").read().strip().rstrip('000')
            elif hardware == 5 :
                temp = open("/sys/bus/acpi/devices/LNXTHERM:00/thermal_zone/temp").read().strip().rstrip('000')
            else:
                return 0

            return float(temp)

        except FloatingPointError:
            return 0
        except ZeroDivisionError:
            return 0
        except ValueError:
            return 0

    def get_product_name(self):
        """
        Example of 'dmidecode' command.

SMBIOS 2.6 present.

Handle 0x0001, DMI type 1, 27 bytes
System Information
	Manufacturer: LG Electronics Inc.
	Product Name: NC1100
	Version: Not Applicable
	Serial Number:
	UUID: 97013E00-653E-1000-99CA-203E01975E65
	Wake-up Type: Power Switch
	SKU Number: Not Specified
	Family: Not Specified

        :return: Product name
        """
        dmi_result = Popen(["dmidecode", "--type", "1"], shell=False, stdout=PIPE)
        std_out, std_err = dmi_result.communicate()
        lines = std_out.split("\n")
        for line in lines:
            if "Product Name" in line:
                return line.split(":")[1].strip()
        return ""

    def get_serial_number(self):
        dmi_result = Popen(["dmidecode", "--type", "1"], shell=False, stdout=PIPE)

        std_out, std_err = dmi_result.communicate()
        lines = std_out.split("\n")
        prop = dict()

        for line in lines:
            if ":" not in line:
                continue

            k, v = line.split(":")

            prop[k.strip()] = v.strip()

        if 'Serial number' in prop and prop['Serial number'] != '':
            return prop['Serial number']
        elif 'UUID' in prop and prop['UUID'] != '':
            # If 'Serial number' is empty, use UUID instead.
            return prop['UUID']

        return ""

    def get_network(self):
        # for ip address
        network_info = Popen("%s/network.sh" % current_dir, shell=False, stdout=PIPE)
        std_out, std_err = network_info.communicate()
        result = dict()
        data = dict()
        data = std_out.split("\n")

        result['ip'] = data[0]
        result['netmask'] = data[1]
        result['gateway'] = data[2]
        result['mac'] = data[3]
        result['dns'] = ""

        return result

    def get_distribution(self):
        dist = platform.dist()

        return "%s %s %s" % (dist[0], dist[1], platform.machine())

    def get_uptime(self):
        uptime = Popen("uptime", shell=False, stdout=PIPE)
        out, err = uptime.communicate()

        if "days" in out:
            # If out has a 'days', it consist of '19:25:41 up 75 days,  4:28,  3 users, ..'.
            up_pos = out.index("up")
            end_pos = out.index(",", up_pos + 3)
            end_pos = out.index(",", end_pos + 1)
        else:
            # If out doesn't has a 'days', it consist of '19:18  up  3:37, 7 users, ..'
            up_pos = out.index("up")
            end_pos = out.index(",", up_pos + 3)

        return out[up_pos + 3: end_pos].strip()

    def get_application_status(self, app_path):
        process1 = Popen(["ps"], stdout=PIPE)
        process2 = Popen(["grep", "-v", "grep"], stdin=process1.stdout, stdout=PIPE)
        ps = Popen(["egrep", "\"%s$\"" % app_path], stdin=process2.stdout, stdout=PIPE)

        out, err = ps.communicate()[0]

        if out != '' and app_path in out:
            return 'running'
        else:
            return 'stop'

    def launch_application(self, app_path):
        if app_path == '':
            return False

        #devnull = open(os.devnull, 'wb')
        # Popen(['nohup', app_path], shell=True, stdout=devnull, stderr=devnull)

        # os.spawnl(os.P_NOWAIT, app_path)

        try:
            pid = os.fork()
        except OSError, e:
            return False

        if pid == 0:
            args = app_path.split()
            os.execv(args[0], args)

        return True

    def stop_application(self, app_path):
        if app_path == '':
            return False

        process1 = Popen(["ps", "ax"], stdout=PIPE)
        process2 = Popen(["grep", "-v", "grep"], stdin=process1.stdout, stdout=PIPE)
        ps = Popen(["egrep", "\"%s$\"" % app_path], stdin=process2.stdout, stdout=PIPE)
        out, err = ps.communicate()[0]

        #signal.signal(signal.SIGCHLD, lambda _x,_y: os.wait())

        try:

            pid = int(out.split()[0])

            #os.system("kill -9 %d" % pid)
            os.kill(pid, 9)
            os.wait()

            return True

        except ValueError:
            return False
        except:
            print("Stop application failed.")
            return False

    def screen_capture(self, capture_path):
        cmd = "%s/%s %s" % (current_dir, FBGRAB, capture_path)
        os.system(cmd)

        return True

    def read_screen_capture(self, capture_path):
        capture_image = ""

        try:
            f = open(capture_path, 'r')

            while True:
                buf = f.read(8192)
                if buf == '':
                    break

                capture_image += buf

            encoded = base64.b64encode(capture_image)

            return encoded
        except IOError, e:
            return ""


class EnvMac(Env):
    def get_kernel_version(self):
        return os.uname()[2]

    def get_cpu_usage(self):
        """
        Result of iostat

      cpu     load average
 us sy id   1m   5m   15m
  4  2 94  1.21 1.25 0.92
        """
        cpu_result = Popen(["iostat", "-n0"], shell=False, stdout=PIPE)
        std_out, str_err = cpu_result.communicate()

        lines = std_out.split("\n")
        # nums: [user, 1sys, idle, 1m, 5m , 15]
        nums = lines[2].strip().split()

        return 100 - int(nums[2])

    def get_free_memory(self):
        total_mem_res = Popen(["sysctl -n hw.memsize"], shell=False, stdout=PIPE)
        out, err = total_mem_res.communicate()

        total_mem = int(out)

        free_mem_result = Popen(["vm_stat | grep 'free'"], shell=False, stdout=PIPE)
        out, err = free_mem_result.communicate()

        free_mem = int(out.split("\n")[0].split()[2].split(".")[0]) * 4096

        return "%.2f" % (float(free_mem) / float(total_mem) * 100.0)

    def get_free_hdd(self):
        hdd_result = Popen(["df", "/"], shell=False, stdout=PIPE)
        out, err = hdd_result.communicate()

        line = out.split("\n")[1]
        free_hdd = line.split()[4].split("%")[0]

        return free_hdd

    def get_distribution(self):
        mac_ver = platform.mac_ver()

        return "MAC OS X %s %s" % (mac_ver[0], mac_ver[-1])


class EnvWindows(Env):
    def get_kernel_version(self):
        return platform.uname()[3]

    def restart_mediaplayer(self):
        time = 3
        restart = 'shutdown -r -t %d' % time
        os.system(restart)

        restart_xpe = 'xpepm -restart'
        os.system(restart_xpe)

    def get_physical_total_memory(self):
        mem = psutil.virtual_memory()
        return str(int(mem.total) / (1024*1024))

    def get_physical_total_hdd(self):
        disk = psutil.disk_usage('/')
        return str(int(disk.total) / (1024*1024))

    def get_os_version(self):
        os_ver = sys.getwindowsversion()

        sys_info = dict()

        sys_info['major'] = os_ver.major
        sys_info['minor'] = os_ver.minor
        sys_info['build'] = os_ver.build
        sys_info['platform'] = os_ver.platform
        sys_info['service_pack'] = os_ver.service_pack
        sys_info['system'] = platform.system()
        sys_info['release'] = platform.release()

        return sys_info

    def get_system_processor(self):
        info = cpuinfo.get_cpu_info()
        return str(info['brand'])

    def get_system_info(self):
        std_out = open('sysInfo.txt', 'r')

        sys_info = dict()

        while 1:
            line = std_out.readline()
            if not line:
                break

            r_state = 1
            if "BIOS Information" in line:
                while r_state is 1:
                    line = std_out.readline()
                    if not line:
                        break

                    if "Vendor" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "Version" in line:
                        sys_info['BIOS Version'] = line.split(":")[1].strip()
                    elif "Release Date" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                        r_state = 0

            if "System Information" in line:
                while r_state is 1:
                    line = std_out.readline()
                    if not line:
                        break

                    if "Manufacturer" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "Product Name" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "Version" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "Serial Number" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "UUID" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "Wake-up Type" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "SKU Number" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                    elif "Family" in line:
                        sys_info[line.split(":")[0].strip()] = line.split(":")[1].strip()
                        r_state = 0

        std_out.close()
        return sys_info


    def get_product_name(self):
        """
        Example of 'dmidecode' command.

SMBIOS 2.6 present.

Handle 0x0001, DMI type 1, 27 bytes
System Information
	Manufacturer: LG Electronics Inc.
	Product Name: NC1100
	Version: Not Applicable
	Serial Number:
	UUID: 97013E00-653E-1000-99CA-203E01975E65
	Wake-up Type: Power Switch
	SKU Number: Not Specified
	Family: Not Specified

        :return: Product name
        """
#        dmi_result = Popen(['dmidecode', '--type', '1'], shell=False, stdout=PIPE)
#        std_out, std_err = dmi_result.communicate()

        std_out = open('sysInfo.txt', 'r')

        while 1:
            line = std_out.readline()
            if not line:
                break
            if "Product Name" in line:
                std_out.close()
                return line.split(":")[1].strip()

#        lines = std_out.split("\n")

#        for line in lines:
#            if "Product Name" in line:
#                return line.split(":")[1].strip()

        std_out.close()
        return ""

    def get_serial_number(self):
#        dmi_result = Popen(['dmidecode', '--type', '1'], shell=False, stdout=PIPE)
#        std_out, std_err = dmi_result.communicate()
        std_out = open('sysInfo.txt', 'r')

        while 1:
            line = std_out.readline()
            if not line:
                break
            if "Serial Number" in line:
                std_out.close()
                return line.split(":")[1].strip()

        std_out.close()
        return ""


    def get_hw_address(self):
        return get_mac()

    def get_network(self):
        result = dict()
        adapter_list_key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE,
                                           r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\NetworkCards')
        adapter_count = _winreg.QueryInfoKey(adapter_list_key)[0]

        for i in xrange(adapter_count):
            try :
                sub_key_name = _winreg.EnumKey(adapter_list_key, i)
                adapter_key = _winreg.OpenKey(adapter_list_key, sub_key_name)
                (adapter_service_name, _) = _winreg.QueryValueEx(adapter_key, "ServiceName")
                (description, _) = _winreg.QueryValueEx(adapter_key, "Description")

                # try:
                adapter_registry_path = os.path.join(
                    r'SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces',
                    adapter_service_name)
                adapter_service_key = _winreg.OpenKey(
                    _winreg.HKEY_LOCAL_MACHINE,
                    adapter_registry_path
                )
            except:
                print('Error continue')
                continue

            try:
                _winreg.QueryValueEx(adapter_service_key, "DhcpSubnetMask")
                isDHCP = True
            except:
                isDHCP = False

            try:
                if isDHCP is True:
                    # SubnetMask
                    [subnet_mask, _] = _winreg.QueryValueEx(adapter_service_key, "DhcpSubnetMask")
                    # IpAddress
                    [ip_address, _] = _winreg.QueryValueEx(adapter_service_key, "DhcpIpAddress")
                    # DefaultGateway
                    [gateway, _] = _winreg.QueryValueEx(adapter_service_key, "DhcpDefaultGateway")
                    # NameServer
                    (dns, _) = _winreg.QueryValueEx(adapter_service_key, "DhcpNameServer")

                if isDHCP is False:
                    # SubnetMask
                    [subnet_mask, _] = _winreg.QueryValueEx(adapter_service_key, "SubnetMask")
                    # IpAddress
                    [ip_address, _] = _winreg.QueryValueEx(adapter_service_key, "IpAddress")
                    # DefaultGateway
                    [gateway, _] = _winreg.QueryValueEx(adapter_service_key, "DefaultGateway")
                    # NameServer
                    (dns, _) = _winreg.QueryValueEx(adapter_service_key, "NameServer")

            except WindowsError as e:
                print('Not valid interface', e)
                continue

            try:
                mac = hex(get_mac()).upper()
                result['interface'] = description.encode('utf-8')

                if len(mac) < 14:
                    result['mac'] = "00:"+mac[2]+mac[3]+":"+mac[4]+mac[5]+":"+mac[6]+mac[7]+":"+mac[8]+mac[9]+":"+mac[10]+mac[11]
                else:
                    result['mac'] = mac[2]+mac[3]+":"+mac[4]+mac[5]+":"+mac[6]+mac[7]+":"+mac[8]+mac[9]+":"+mac[10]+mac[11]+\
                                    ":"+mac[12]+mac[13]

            except:
                print('interface & mac parsing error.')

            try:
                result['ip'] = ip_address[0].encode('utf-8')
                if len(result['ip']) == 1:
                    result['ip'] = ip_address.encode('utf-8')
                print result['ip']

                result['netmask'] = subnet_mask[0].encode('utf-8')
                if len(result['netmask']) == 1:
                    result['netmask'] = subnet_mask.encode('utf-8')
                print result['netmask']

                result['gateway'] = gateway[0].encode('utf-8')
                if len(result['gateway']) == 1:
                    result['gateway'] = gateway.encode('utf-8')

                result['dns'] = dns[0].encode('utf-8')
                if len(result['dns']) == 1:
                    result['dns'] = dns.encode('utf-8')
            except:
                print('getting network information failed.')

        return result

    def get_cpu_usage(self):
        try:
            return psutil.cpu_percent()
        except RuntimeError:
            return 0.0
        except ValueError:
            return 0.0

    def get_memory_usage(self):
        try:
            return psutil.phymem_usage().percent
        except RuntimeError:
            return 0.0
        except ValueError:
            return 0.0

    def get_hdd_usage(self):
        try:
            return psutil.disk_usage('/').percent
        except RuntimeError:
            return 0.0
        except ValueError:
            return 0.0

    def get_free_memory(self):
        try:
            return 100.0 - psutil.phymem_usage().percent
        except RuntimeError:
            return 0.0
        except ValueError:
            return 0.0

    def get_free_hdd(self):
        try:
            return 100.0 - psutil.disk_usage('/').percent
        except RuntimeError:
            return 0.0
        except ValueError:
            return 0.0

    def get_distribution(self):
        return platform.platform()

    def get_cpu_temp(self, hardware=None):
        hwinfo_parser = ConfigParser()

        try:
            data_set = hwinfo_parser.read("hwinfo.ini")
            if len(data_set) == 0:
                #self.logger.error("Open 'hwinfo.ini' file failed." )
                return 0

            cpu_info = hwinfo_parser.get('HWINFO', 'CPU')
            return int(float(cpu_info))
        except:
            return 0

    def get_hdd_temp(self):
        hwinfo_parser = ConfigParser()

        try:
            data_set = hwinfo_parser.read("hwinfo.ini")
            if len(data_set) == 0:
                #self.logger.error("Open 'hwinfo.ini' file failed." )
                return 0

            hdd_info = hwinfo_parser.get('HWINFO', 'HDD')
            return int(float(hdd_info))
        except:
            return 0

    def get_fan_speed(self):
        hwinfo_parser = ConfigParser()

        try:
            data_set = hwinfo_parser.read("hwinfo.ini")
            if len(data_set) == 0:
                #self.logger.error("Open 'hwinfo.ini' file failed." )
                return 0

            fan_info = hwinfo_parser.get('HWINFO', 'FAN')
            return int(float(fan_info))
        except:
            return 0

    def read_from_file(self, filename, section, required_props=None):
        config_parser = ConfigParser()
        data = dict()

        try:
            data_set = config_parser.read(filename)
            if len(data_set) == 0:
                self.logger.error("Open '%s' file failed." % (filename))
                return None

            if section not in config_parser.sections():
                return dict()

            for k, v in config_parser.items(section):
                data[k] = v

            if required_props is not None:
                for prop in required_props:
                    if prop not in data:
                        self.logger.error("%s must have %s property." % (filename, prop))
                        return None

            return data

        except IOError, e:
            self.logger.error("Open '%s' file failed: %s" % (filename, str(e)))
            return None

    def get_uptime(self):
        from uptime import uptime

        up = uptime()
        if up is not None:
            parts = []

            days, up = up // 86400, up % 86400
            if days is 0:
                parts.append('%d day' % 0)
            else:
                parts.append('%d day%s' % (days, 's' if days != 0 and days != 1 else ''))

            hours, up = up // 3600, up % 3600
            if hours is 0:
                parts.append(' %02d:' % 0)
            else:
                parts.append(' %02d:' % hours)

            minutes, up = up // 60, up % 60
            if minutes is 0:
                parts.append('%02d' % 0)
            else:
                parts.append('%02d' % minutes)

            uptime = '%s' % ''.join(parts)

        return uptime

    def find_pid(self, name):
        task = os.popen2('tasklist /FI "IMAGENAME eq '+name+'"')
        try:
            info = task[1].readlines()[3].split()
            #print info
        except:
            info = [name, "NotFound"]
        return info[1]

    def get_application_status(self, app_path):
        app_name = os.path.basename(app_path).split()[0]

        dic = []
        if app_name == '':
            return dic

        plist = psutil.pids()
        for pn in plist:

            try:
                p = psutil.Process(pn)

                if app_name.lower() == p.name().lower():
                    pid_value = self.find_pid(app_name.lower())
                    #print ' --> ', pid_value

                    dic.append(p.status())
                    dic.append(psutil.Process(pid=int(pid_value)).cpu_percent())
                    dic.append(psutil.Process(pid=int(pid_value)).memory_percent())
                    return dic
            except:
                pass

        dic.insert(0, 'stop')
        return dic

    # Traffic
    def get_network_traffic(self, traffic, time):
        return traffic / time

    def checksum(self, source_string):
        """
        I'm not too confident that this is right but testing seems
        to suggest that it gives the same answers as in_cksum in ping.c
        """
        sum = 0
        count_to = (len(source_string) / 2) * 2
        for count in xrange(0, count_to, 2):
            this = ord(source_string[count + 1]) * 256 + ord(source_string[count])
            sum = sum + this
            sum = sum & 0xffffffff # Necessary?

        if count_to < len(source_string):
            sum = sum + ord(source_string[len(source_string) - 1])
            sum = sum & 0xffffffff # Necessary?

        sum = (sum >> 16) + (sum & 0xffff)
        sum = sum + (sum >> 16)
        answer = ~sum
        answer = answer & 0xffff

        # Swap bytes. Bugger me if I know why.
        answer = answer >> 8 | (answer << 8 & 0xff00)

        return answer

    def receive_one_ping(self, my_socket, id, timeout):
        """
        Receive the ping from the socket.
        """
        time_left = timeout
        while True:
            started_select = time.time()
            what_ready = select.select([my_socket], [], [], time_left)
            how_long_in_select = (time.time() - started_select)
            if what_ready[0] == []: # Timeout
                return

            time_received = time.time()
            received_packet, addr = my_socket.recvfrom(1024)
            icmpHeader = received_packet[20:28]
            type, code, checksum, packet_id, sequence = struct.unpack(
                "bbHHh", icmpHeader
            )
            if packet_id == id:
                bytes = struct.calcsize("d")
                time_sent = struct.unpack("d", received_packet[28:28 + bytes])[0]
                return time_received - time_sent

            time_left = time_left - how_long_in_select
            if time_left <= 0:
                return

    def send_one_ping(self, my_socket, dest_addr, id, psize):
        """
        Send one ping to the given >dest_addr<.
        """
        dest_addr  =  socket.gethostbyname(dest_addr)

        # Remove header size from packet size
        psize = psize - 8

        # Header is type (8), code (8), checksum (16), id (16), sequence (16)
        my_checksum = 0

        # Make a dummy heder with a 0 checksum.
        header = struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, my_checksum, id, 1)
        bytes = struct.calcsize("d")
        data = (psize - bytes) * "Q"
        data = struct.pack("d", time.time()) + data

        # Calculate the checksum on the data and the dummy header.
        my_checksum = self.checksum(header + data)

        # Now that we have the right checksum, we put that in. It's just easier
        # to make up a new header than to stuff it into the dummy.
        header = struct.pack(
            "bbHHh", ICMP_ECHO_REQUEST, 0, socket.htons(my_checksum), id, 1
        )
        packet = header + data
        my_socket.sendto(packet, (dest_addr, 1)) # Don't know about the 1

    def do_one(self, dest_addr, timeout, psize):
        """
        Returns either the delay (in seconds) or none on timeout.
        """
        icmp = socket.getprotobyname("icmp")
        try:
            my_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, icmp)

        except socket.error, (errno, msg):
            if errno == 1:
                # Operation not permitted
                msg = msg + (
                    " - Note that ICMP messages can only be sent from processes"
                    " running as root."
                )
                raise socket.error(msg)
            raise # raise the original error

        my_id = os.getpid() & 0xFFFF
        self.send_one_ping(my_socket, dest_addr, my_id, psize)
        delay = self.receive_one_ping(my_socket, my_id, timeout)

        my_socket.close()
        return delay

    def verbose_ping(self, dest_addr, timeout=2, psize=64):
        """
        Send `count' ping with `psize' size to `dest_addr' with
        the given `timeout' and display the result.
        """
        print "ping %s with ..." % dest_addr,
        try:
            delay = self.do_one(dest_addr, timeout, psize)
        except socket.gaierror, e:
            return -1

        if delay is None:
            return -1
        else:
            return delay * 1000

    def get_response_time(self, ping_address):
        result = self.verbose_ping(ping_address)
        return result

    def launch_application(self, app_path):
        if app_path == '':
            return False

        print app_path

        app_name_with_param = app_path.split()

        print app_name_with_param

        app_result = Popen(app_name_with_param, shell=False, stdout=PIPE)

        return True

    def stop_application(self, app_path):
        if app_path == '':
            return False

        app_name = os.path.basename(app_path).split()[0]

        if app_name == '':
            return 'stop'

        plist = psutil.pids()
        for pn in plist:
            try:
                p = psutil.Process(pn)
                if app_name.lower() == p.name().lower():
                    p.kill()
            except:
                pass

    def screen_capture(self, capture_path):
        from PIL import ImageGrab
        try:
            img = ImageGrab.grab()
            current_width = img.size[0]
            current_height = img.size[1]
            if current_width >= current_height:
                if current_width < 1024:
                    base_width = current_width
                else:
                    base_width = 1024
                #print ("0....................current_width : " + str(current_width) + " current_height : " + str(current_height) + " base_width : " + str(base_width) )
                wpercent = (base_width/float(current_width))
                hsize = int((float(current_height)*float(wpercent)))
                #print ("1....................wpercent : " + str(wpercent) + " hsize : " + str(hsize))
                img = img.resize((base_width, hsize))
                img.save(capture_path, "JPEG", quality=50)
                #print ("1....................capture_path : " + capture_path)
                if current_width < 138:
                    base_width = current_width
                else:
                    base_width = 138
                wpercent = (base_width/float(current_width))
                hsize = int((float(current_height)*float(wpercent)))
                #print ("2...................wpercent : " + str(wpercent) + " hsize : " + str(hsize))
                img = img.resize((base_width, hsize))
                capture_path = capture_path.replace("_large.jpg", ".jpg")
                #print ("2....................capture_path : " + capture_path)
                img.save(capture_path, "JPEG", quality=50)
            else:
                if current_height < 768:
                    base_height = current_height
                else:
                    base_height = 768

                wpercent = (base_height/float(current_height))
                wsize = int((float(current_width)*float(wpercent)))
                img = img.resize((wsize, base_height))
                img.save(capture_path, "JPEG", quality=50)
                #print ("3....................capture_path : " + capture_path)
                if current_height < 98:
                    base_height = current_height
                else:
                    base_height = 98

                wpercent = (base_height/float(current_height))
                wsize = int((float(current_width)*float(wpercent)))
                img = img.resize((wsize, base_height))
                capture_path = capture_path.replace("_large.jpg", ".jpg")
                #print ("4....................capture_path : " + capture_path)
                img.save(capture_path, "JPEG", quality=50)
        except:
            self.logger.error("screen_capture is failed.")

        return True
    #def screen_capture(self, capture_path):
    #    from PIL import ImageGrab
    #    try:
    #        img = ImageGrab.grab()
    #        current_width = img.size[0]
    #        current_height = img.size[1]
    #        if current_width >= current_height:
    #            if current_width < 138:
    #                base_width = current_width
    #            else:
    #                base_width = 138
    #            wpercent = (base_width/float(current_width))
    #            hsize = int((float(current_height)*float(wpercent)))
    #            img = img.resize((base_width, hsize))
    #            img.save(capture_path, "JPEG", quality=50)
    #            print ("....................capture_path : " + capture_path)
    #            if current_width < 1024:
    #                base_width = current_width
    #            else:
    #                base_width = 1024
    #            wpercent = (base_width/float(current_width))
    #            hsize = int((float(current_height)*float(wpercent)))
    #            img = img.resize((base_width, hsize))
    #            capture_path.replace(".jpg", "_large.jpg")
    #            print ("....................capture_path : " + capture_path)
    #            img.save(capture_path, "JPEG", quality=50)
    #        else:
    #            if current_height < 98:
    #                base_height = current_height
    #            else:
    #                base_height = 98
    #            wpercent = (base_height/float(current_height))
    #            wsize = int((float(current_width)*float(wpercent)))
    #            img = img.resize((wsize, base_height))
    #            img.save(capture_path, "JPEG", quality=50)

    #            if current_height < 768:
    #                base_height = current_height
    #            else:
    #                base_height = 768
    #            wpercent = (base_height/float(current_height))
    #            wsize = int((float(current_width)*float(wpercent)))
    #            img = img.resize((wsize, base_height))
    #            capture_path.replace(".jpg", "_large.jpg")
    #            img.save(capture_path, "JPEG", quality=50)
    #    except:
    #        self.logger.error("screen_capture is failed.")

    #    return True
    def read_screen_capture(self, capture_path):
        capture_image = ""
        try:
            with open(capture_path, "rb ") as imageFile:
                encoded = base64.b64encode(imageFile.read())

            return encoded
        except IOError, e:
            return ""


if __name__ == '__main__':
    env = Env.get_env().get_network()

