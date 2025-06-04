.. molass_legacy documentation master file, created by
   sphinx-quickstart on Sat Mar 15 11:23:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Molass Legacy Reference
=======================

Molass Legacy is an open-source version of `MOLASS <https://www.jstage.jst.go.jp/article/biophysico/20/1/20_e200001/_article>`_, an analytical tool for SEC-SAXS experiment data currently hosted at `Photon Factory <https://pfwww.kek.jp/saxs/MOLASS.html>`_.

It can be installed via pip as follows::

   pip install molass_legacy

And run from the command line as follows::

   molass

See the User's Guide at the original `MOLASS page <https://pfwww.kek.jp/saxs/MOLASS.html>`_ for more information on how to use it.

Main Function
----------------

.. toctree::
   :maxdepth: 5

   source/molass_legacy.main

Module Functions
----------------

.. automodule:: molass_legacy
   :members:
   :undoc-members:
   :show-inheritance:


Submodules
-----------

.. toctree::
   :maxdepth: 4

   source/molass_legacy.Alsaker
   source/molass_legacy.ATSAS
   source/molass_legacy.AutorgKek
   source/molass_legacy.Baseline
   source/molass_legacy.Batch
   source/molass_legacy.BoundedLRF
   source/molass_legacy.CFSD
   source/molass_legacy.CharFunc
   source/molass_legacy.Conc
   source/molass_legacy.DataStructure
   source/molass_legacy.Decomposer
   source/molass_legacy.DecompProc
   source/molass_legacy.DENSS
   source/molass_legacy.DevelUtils
   source/molass_legacy.Distance
   source/molass_legacy.ExcelProcess
   source/molass_legacy.DMM
   source/molass_legacy.EFA
   source/molass_legacy.Elution
   source/molass_legacy.Env
   source/molass_legacy.Error
   source/molass_legacy.Estimators
   source/molass_legacy.ExcelProcess
   source/molass_legacy.Experiment
   source/molass_legacy.Extrapolation
   source/molass_legacy.Factors
   source/molass_legacy.Global
   source/molass_legacy.Gmm
   source/molass_legacy.GuinierAnalyzer
   source/molass_legacy.GuinierTools
   source/molass_legacy.GuiParts
   source/molass_legacy.HdcTheory
   source/molass_legacy.Hplc
   source/molass_legacy.ICIS
   source/molass_legacy.IFT
   source/molass_legacy.InputProcess
   source/molass_legacy.Irregular
   source/molass_legacy.Jupyter
   source/molass_legacy.KDE
   source/molass_legacy.KekLib
   source/molass_legacy.Kratky
   source/molass_legacy.LoaderProcess
   source/molass_legacy.LRF
   source/molass_legacy.Mapping
   source/molass_legacy.MD
   source/molass_legacy.Menus
   source/molass_legacy.Microfluidics
   source/molass_legacy.ModelParams
   source/molass_legacy.Models
   source/molass_legacy.MultiProc
   source/molass_legacy.MXD
   source/molass_legacy.ObjectiveFunctions
   source/molass_legacy.OnTheFly
   source/molass_legacy.Optimizer
   source/molass_legacy.Particles
   source/molass_legacy.Peaks
   source/molass_legacy.Promegranate
   source/molass_legacy.Protein
   source/molass_legacy.PSD
   source/molass_legacy.Python
   source/molass_legacy.QuickAnalysis
   source/molass_legacy.RangeEditors
   source/molass_legacy.Rank
   source/molass_legacy.ReAtsas
   source/molass_legacy.ret_codes
   source/molass_legacy.Reports
   source/molass_legacy.Rgg
   source/molass_legacy.RgProcess
   source/molass_legacy.Saxs
   source/molass_legacy.SecSaxs
   source/molass_legacy.SecTheory
   source/molass_legacy.SecTools
   source/molass_legacy.Selective
   source/molass_legacy.SerialAnalyzer
   source/molass_legacy.SimTools
   source/molass_legacy.Simulative
   source/molass_legacy.Solvers
   source/molass_legacy.SSDC
   source/molass_legacy.Stochastic
   source/molass_legacy.Sympy
   source/molass_legacy.Synthesizer
   source/molass_legacy.Test
   source/molass_legacy.Trials.
   source/molass_legacy.Trimming
   source/molass_legacy.Tutorials
   source/molass_legacy.UV
   source/molass_legacy.V2PropOptimizer
   source/molass_legacy.Wave


Tool Functions
----------------

.. toctree::
   :maxdepth: 5

   source/EditRst