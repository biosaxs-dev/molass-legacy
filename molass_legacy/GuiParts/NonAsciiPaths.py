"""

    NonAsciiPaths.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""

def are_all_single_bytes(s):
    return len(s.encode()) == len(s)

def nonascii_path_check(path, parent):
    if are_all_single_bytes(path):
        return True

    if not parent.env_info.atsas_is_available:
        # no need to ask
        return True

    parent.set_an_folder_error()
    parent.update()

    import molass_legacy.KekLib.CustomMessageBox as MessageBox
    yes = MessageBox.askyesno("Multibyte Path Option Query",
        '"%s"\n'
        "seems to include multi-byte characters\n"
        "which may not be processed properly in ATSAS programs.\n"
        "You can simply continue without ATSAS programs,\n"
        "or retry after replacing it with another single-byte only character folder.\n"
        "Woud you like to continue simply?" % path,
        parent=parent,
        )
    if yes:
        from Env.EnvInfo import EnvInfo, set_global_env_info
        from molass_legacy._MOLASS.SerialSettings import set_setting
        set_setting('revoke_atsas', 1)
        env_info = EnvInfo()
        env_info.atsas_is_available = False
        parent.tmp_logger.info("ATSAS has been revoked.")
        parent.env_info = env_info
        parent.analyzer.env_info = env_info     # this is problematic. consider better implementation
        set_global_env_info(env_info)
        parent.an_folder_entry.config(fg='black')
        parent.update()
        return True
    else:
        return False

if __name__ == '__main__':
    for path in [r"D:\田中\Data", r"D:\Data"]:
        print(path, are_all_single_bytes(path))
