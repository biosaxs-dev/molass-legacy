"""

    GuiReferences.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF

"""
import os
from molass_legacy.KekLib.OurTkinter     import Tk
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy._MOLASS.Version import get_version_string

def get_doc_folder_url():
    from molass_legacy.KekLib.BasicUtils import get_home_folder
    # home = os.getcwd().replace( '\\', '/' )
    home = get_home_folder()
    url = 'file://' + home + '/doc'
    # print('doc_folder_url=', url)
    return url

browser = None

def get_default_browser():
    global browser
    if browser is None:
        import webbrowser
        browser = webbrowser.get()

    return browser

def get_preferable_browser():
    global browser
    if browser is None:
        import webbrowser
        try:
            # https://stackoverflow.com/questions/48056052/webbrowser-get-could-not-locate-runnable-browser
            chrome_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
            browser = webbrowser.get('chrome')
        except:
            browser = webbrowser.get()

    return browser

class GuiReferencesMenu(Tk.Menu):
    def __init__(self, parent, menubar ):
        self.parent = parent
        doc_folder_url = get_doc_folder_url()
        browser = get_preferable_browser()

        web_references = []

        if get_version_string().find("_MOLASS 2") >= 0:
            web_references += [("_MOLASS V2 Manipulation Guide", doc_folder_url + '/_MOLASS-Manipulation-Guide-2_0-draft.pdf')]

        web_references += [
            ("_MOLASS V1 User's Guide", doc_folder_url + '/_MOLASS-UsersGuide-1_0_13.pdf'),
            ("Formula (2.13), Feigin and Svergun 1987", doc_folder_url + '/InterferenceEffects-feigin-svergun-1987-excerpt.pdf'),
            # ('Automation of Guinier Analysis in SAXS', doc_folder_url + '/AutomationOfGuinierAnalysis.pdf'),
            # ('Bi-component Guinier-Porod Model', doc_folder_url + '/BicomponentGuinierPorodModel.pdf'),
            # ('A new Guinier–Porod model', 'https://www.ncnr.nist.gov/staff/hammouda/publications/2010_hammouda_j_appl_crys.pdf'),
            # ('Gaussian Processes', 'http://scikit-learn.org/stable/modules/gaussian_process.html'),
            ('Linear least squares (mathematics)', 'https://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29#Weighted_linear_least_squares'),
            ('ATSAS AUTORG manual', 'http://www.embl-hamburg.de/biosaxs/manuals/autorg.html'),
            ('ATSAS ALMERGE manual', 'https://www.embl-hamburg.de/biosaxs/manuals/almerge.html'),
            ('ATSAS DATGNOM4 manual', 'https://www.embl-hamburg.de/biosaxs/manuals/datgnom.html'),
            ]

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade( label="References", menu=self )
        for label_, url_ in web_references:
            self.add_command( label=label_, command=lambda url=url_: browser.open_new( url ) )
