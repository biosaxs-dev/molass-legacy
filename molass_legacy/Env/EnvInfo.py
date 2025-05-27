"""
    EnvInfo.py

    Copyright (c) 2019-2022, SAXS Team, KEK-PF
"""
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.MachineTypes import get_bootup_state

env_info = None

def set_global_env_info(info):
    global env_info
    env_info = info

def get_global_env_info(gpu_info=False):
    global env_info
    if env_info is None:
        env_info = EnvInfo()
    if gpu_info:
        env_info.set_gpu_availability()
    return env_info

class EnvInfo:
    def __init__(self):
        bootup_state = get_bootup_state()
        self.normal = bootup_state.find("Normal") >= 0
        self.set_atsasl_availability()
        self.set_excel_availability()
        self.gpu_info = None               # self.set_gpu_availability() will be delayed
        self.latex_is_available = None

    def set_atsasl_availability(self):
        revoke_atsas = get_setting('revoke_atsas')
        if revoke_atsas:
            self.atsas_is_available = False
            return

        try:
            from molass_legacy.ATSAS.AtsasUtils import get_version
            self.atsas_version = get_version()
            self.atsas_is_available = self.atsas_version is not None
        except:
            # TODO: handle this better
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)
            self.atsas_is_available = False

    def set_excel_availability(self):
        from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
        revoke_excel = get_setting('revoke_excel')
        if not self.normal or revoke_excel:
            self.excel_version = None
            self.excel_is_available = False
            self.com_error = None
            return

        try:
            from molass_legacy.KekLib.ExcelCOM import excel_availability
            self.excel_version = excel_availability()
            self.excel_is_available = self.excel_version is not None
            com_error = None
        except:
            com_error = str(ExceptionTracebacker())
            self.excel_is_available = False
        """
        set_setting('excel_is_available', self.excel_is_available)
            using persistent memory through SerialSettings to send info to the execution thread
            seems unreliable.
            
        """
        self.com_error = com_error

    def set_gpu_availability(self):
        revoke_cuda = get_setting('revoke_cuda')
        if revoke_cuda:
            self.cuda_tools_version = None
            self.nvidiagpu_is_available = False
            return

        if self.gpu_info is None:
            from .GpuInfo import GpuInfo
            self.gpu_info = gpu_info = GpuInfo()
            self.cuda_tools_version = gpu_info.cupy_version, gpu_info.cuda_version
            self.nvidiagpu_is_available = gpu_info.cupy_ok()
            if self.nvidiagpu_is_available:
                self.cuda_tools_ver_str = '(cupy %s, cuda %s)' % self.cuda_tools_version[0:2]
            else:
                self.cuda_tools_ver_str = ''

    def get_gpu_reason_texts(self):
        return self.gpu_info.get_reason_texts()

    def show_and_log_if_unavailable(self, dialog, logger):
        com_error = self.com_error
        if self.excel_is_available:
            ver_ = ' %s' % self.excel_version
            not_ = ''
        else:
            ver_ = ''
            not_ = 'not '
        excel_availability_message = 'Excel%s is %savailable.' % (ver_, not_)
        if com_error is not None:
            excel_availability_message += " com_error=" + com_error
        if self.excel_is_available:
            logger.info(excel_availability_message)
        else:
            logger.warning(excel_availability_message)
            if dialog is not None and get_setting('no_excel_warning'):
                from molass_legacy.SerialAnalyzer.NoExcelWarningDialog import NoExcelWarningDialog
                warn_dialog = NoExcelWarningDialog(dialog, com_error)
                warn_dialog.show()

        if self.atsas_is_available:
            ver_ = ' ' + self.atsas_version
            not_ = ''
        else:
            ver_ = '' 
            not_ = 'not '
        atsas_availability_message = 'ATSAS%s is %savailable.' % (ver_, not_)
        if self.atsas_is_available:
            logger.info(atsas_availability_message)
        else:
            logger.warning(atsas_availability_message)

        if self.gpu_info is not None:
            if self.nvidiagpu_is_available:
                cuda_ver = ' ' + self.cuda_tools_ver_str
                not_ = ''
            else:
                cuda_ver = ''
                not_ = 'not '
            nvidiagpu_availability_message = 'NVIDIA GPU%s is %savailable.' % (cuda_ver, not_)
            if self.nvidiagpu_is_available:
                logger.info(nvidiagpu_availability_message)
            else:
                logger.warning(nvidiagpu_availability_message)

    def get_latex_availability(self):
        if self.latex_is_available is None:
            try:
                import subprocess
                ret = subprocess.run(['latex', '-version'])
                self.latex_is_available = ret.returncode == 0
            except:
                self.latex_is_available = False
        return self.latex_is_available
