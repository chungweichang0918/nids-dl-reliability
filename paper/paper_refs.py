"""參考文獻 (IEEE 格式)。每一筆都已查證作者、出處、卷期與頁碼；沒有頁碼的是出處未提供。"""

REFS = {
    # --- 2026-11-18 補充的近年研究: 每篇都在出版社、DOI、arXiv 或作者頁面上確認存在；
    #     只在搜尋引擎索引看到、未在開啟的頁面上確認的頁碼一律不列。
    "flood2024": "R. Flood, G. Engelen, D. Aspinall, and L. Desmet, “Bad design smells in benchmark NIDS "
                 "datasets,” in *Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P)*, 2024, pp. 658–675, "
                 "doi: 10.1109/EuroSP60621.2024.00042.",
    "lanvin2022": "M. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Mé, and E. Totel, “Errors in the CICIDS2017 "
                  "dataset and the significant differences in detection performances it makes,” in *Proc. Int. Conf. "
                  "Risks Secur. Internet Syst. (CRiSIS)*, LNCS, Springer, 2022, pp. 18–33, "
                  "doi: 10.1007/978-3-031-31108-6_2.",
    "dhooge2022": "L. D’hooge, M. Verkerken, B. Volckaert, T. Wauters, and F. De Turck, “Establishing the "
                  "contaminating effect of metadata feature inclusion in machine-learned network intrusion detection "
                  "models,” in *Proc. Detection of Intrusions and Malware, and Vulnerability Assessment (DIMVA)*, "
                  "LNCS 13358, 2022, doi: 10.1007/978-3-031-09484-2_2.",
    "goldschmidt2025": "P. Goldschmidt and D. Chudá, “Network intrusion datasets: A survey, limitations, and "
                       "recommendations,” *Comput. Secur.*, vol. 156, Art. no. 104510, 2025, "
                       "doi: 10.1016/j.cose.2025.104510.",
    "cantone2024": "M. Cantone, C. Marrocco, and A. Bria, “Machine learning in network intrusion detection: A "
                   "cross-dataset generalization study,” *IEEE Access*, vol. 12, pp. 144489–144508, 2024, "
                   "doi: 10.1109/ACCESS.2024.3472907.",
    "apruzzese2022xeval": "G. Apruzzese, L. Pajola, and M. Conti, “The cross-evaluation of machine learning-based "
                          "network intrusion detection systems,” *IEEE Trans. Netw. Serv. Manag.*, vol. 19, no. 4, "
                          "pp. 5152–5169, 2022, doi: 10.1109/TNSM.2022.3157344.",
    "sarhan2022": "M. Sarhan, S. Layeghy, and M. Portmann, “Evaluating standard feature sets towards increased "
                  "generalisability and explainability of ML-based network intrusion detection,” *Big Data Res.*, "
                  "2022, doi: 10.1016/j.bdr.2022.100359.",
    "kus2022": "D. Kus et al., “A false sense of security? Revisiting the state of machine learning-based industrial "
               "intrusion detection,” in *Proc. ACM Cyber-Physical Syst. Secur. Workshop (CPSS)*, 2022, "
               "doi: 10.1145/3494107.3522773.",
    "apruzzese2023sok": "G. Apruzzese, P. Laskov, and J. Schneider, “SoK: Pragmatic assessment of machine learning "
                        "for network intrusion detection,” in *Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P)*, "
                        "2023, doi: 10.1109/EuroSP57164.2023.00042.",
    "schrotter2024": "M. Schrötter, A. Niemann, and B. Schnor, “A comparison of neural-network-based intrusion "
                     "detection against signature-based detection in IoT networks,” *Information*, vol. 15, no. 3, "
                     "Art. no. 164, 2024, doi: 10.3390/info15030164.",
    "sheatsley2022": "R. Sheatsley, N. Papernot, M. Weisman, G. Verma, and P. McDaniel, “Adversarial examples for "
                     "network intrusion detection systems,” *J. Comput. Secur.*, 2022, doi: 10.3233/JCS-210094.",
    "vitorino2023": "J. Vitorino, I. Praça, and E. Maia, “Towards adversarial realism and robust learning for IoT "
                    "intrusion detection and classification,” *Ann. Telecommun.*, 2023, "
                    "doi: 10.1007/s12243-023-00953-y.",
    "debicha2023": "I. Debicha, B. Cochez, T. Kenaza, T. Debatty, J.-M. Dricot, and W. Mees, “Adv-Bot: Realistic "
                   "adversarial botnet attacks against network intrusion detection systems,” *Comput. Secur.*, "
                   "Art. no. 103176, 2023, doi: 10.1016/j.cose.2023.103176.",
    "apruzzese2024drift": "G. Apruzzese, A. Fass, and F. Pierazzi, “When adversarial perturbations meet concept "
                          "drift: An exploratory analysis on ML-NIDS,” in *Proc. ACM Workshop Artif. Intell. Secur. "
                          "(AISec)*, 2024, doi: 10.1145/3689932.3694757.",
    "elsayed2024": "S. Elsayed, K. Mohamed, and M. A. Madkour, “A comparative study of using deep learning "
                   "algorithms in network intrusion detection,” *IEEE Access*, vol. 12, pp. 58851–58870, 2024, "
                   "doi: 10.1109/ACCESS.2024.3389096.",
    "tavallaee2009": "M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 "
                     "data set,” in *Proc. IEEE Symp. Comput. Intell. Secur. Defense Appl. (CISDA)*, 2009, pp. 1–6.",
    "moustafa2015": "N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection "
                    "systems (UNSW-NB15 network data set),” in *Proc. Military Commun. Inf. Syst. Conf. (MilCIS)*, "
                    "2015, pp. 1–6.",
    "moustafa2016": "N. Moustafa and J. Slay, “The evaluation of network anomaly detection systems: Statistical "
                    "analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set,” *Inf. Secur. J.: "
                    "A Global Perspective*, vol. 25, no. 1–3, pp. 18–31, 2016.",
    "sharafaldin2018": "I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward generating a new intrusion "
                       "detection dataset and intrusion traffic characterization,” in *Proc. Int. Conf. Inf. Syst. "
                       "Secur. Privacy (ICISSP)*, 2018, pp. 108–116, doi: 10.5220/0006639801080116.",
    "engelen2021": "G. Engelen, V. Rimmer, and W. Joosen, “Troubleshooting an intrusion detection dataset: The "
                   "CICIDS2017 case study,” in *Proc. IEEE Secur. Privacy Workshops (SPW)*, 2021, pp. 7–12.",
    "liu2022": "L. Liu, G. Engelen, T. Lynar, D. Essam, and W. Joosen, “Error prevalence in NIDS datasets: A case "
               "study on CIC-IDS-2017 and CSE-CIC-IDS-2018,” in *Proc. IEEE Conf. Commun. Netw. Secur. (CNS)*, "
               "2022, pp. 254–262, doi: 10.1109/CNS56114.2022.9947235.",
    "mchugh2000": "J. McHugh, “Testing intrusion detection systems: A critique of the 1998 and 1999 DARPA intrusion "
                  "detection system evaluations as performed by Lincoln Laboratory,” *ACM Trans. Inf. Syst. Secur.*, "
                  "vol. 3, no. 4, pp. 262–294, 2000.",
    "sommer2010": "R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network "
                  "intrusion detection,” in *Proc. IEEE Symp. Secur. Privacy*, 2010, pp. 305–316.",
    "arp2022": "D. Arp et al., “Dos and don’ts of machine learning in computer security,” in *Proc. USENIX Secur. "
               "Symp.*, 2022, pp. 3971–3988.",
    "yin2017": "C. Yin, Y. Zhu, J. Fei, and X. He, “A deep learning approach for intrusion detection using recurrent "
               "neural networks,” *IEEE Access*, vol. 5, pp. 21954–21961, 2017, doi: 10.1109/ACCESS.2017.2762418.",
    "vinayakumar2019": "R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, and "
                       "S. Venkatraman, “Deep learning approach for intelligent intrusion detection system,” "
                       "*IEEE Access*, vol. 7, pp. 41525–41550, 2019, doi: 10.1109/ACCESS.2019.2895334.",
    "ferrag2020": "M. A. Ferrag, L. Maglaras, S. Moschoyiannis, and H. Janicke, “Deep learning for cyber security "
                  "intrusion detection: Approaches, datasets, and comparative study,” *J. Inf. Secur. Appl.*, "
                  "vol. 50, Art. no. 102419, 2020.",
    "gorishniy2021": "Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for "
                     "tabular data,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021.",
    "grinsztajn2022": "L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep "
                      "learning on typical tabular data?” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), "
                      "Datasets and Benchmarks Track*, 2022.",
    "shwartz2022": "R. Shwartz-Ziv and A. Armon, “Tabular data: Deep learning is not all you need,” *Inf. Fusion*, "
                   "vol. 81, pp. 84–90, 2022, doi: 10.1016/j.inffus.2021.11.011.",
    "breiman2001": "L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.",
    "chen2016": "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD "
                "Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.",
    "bouthillier2019": "X. Bouthillier, C. Laurent, and P. Vincent, “Unreproducible research is reproducible,” in "
                       "*Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 725–734.",
    "pineau2021": "J. Pineau et al., “Improving reproducibility in machine learning research,” *J. Mach. Learn. "
                  "Res.*, vol. 22, no. 164, pp. 1–20, 2021.",
    "reimers2017": "N. Reimers and I. Gurevych, “Reporting score distributions makes a difference: Performance study "
                   "of LSTM-networks for sequence tagging,” in *Proc. Conf. Empir. Methods Nat. Lang. Process. "
                   "(EMNLP)*, 2017, pp. 338–348.",
    "bergstra2012": "J. Bergstra and Y. Bengio, “Random search for hyper-parameter optimization,” *J. Mach. Learn. "
                    "Res.*, vol. 13, pp. 281–305, 2012.",
    "hochreiter1997": "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, "
                      "pp. 1735–1780, 1997.",
    "cho2014": "K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine "
               "translation,” in *Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP)*, 2014, pp. 1724–1734.",
    "bengio1994": "Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term dependencies with gradient descent is "
                  "difficult,” *IEEE Trans. Neural Netw.*, vol. 5, no. 2, pp. 157–166, 1994.",
    "pascanu2013": "R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” "
                   "in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2013, pp. 1310–1318.",
    "kingma2015": "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. Int. Conf. Learn. "
                  "Represent. (ICLR)*, 2015.",
    "chollet2015": "F. Chollet et al., “Keras,” 2015. [Online]. Available: https://keras.io",
    "paszke2019": "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in "
                  "*Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8024–8035.",
    "omalley2019": "T. O’Malley et al., “KerasTuner,” 2019. [Online]. Available: "
                   "https://github.com/keras-team/keras-tuner",
    "welch1947": "B. L. Welch, “The generalization of ‘Student’s’ problem when several different population "
                 "variances are involved,” *Biometrika*, vol. 34, no. 1–2, pp. 28–35, 1947.",
    "holm1979": "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, "
                "no. 2, pp. 65–70, 1979.",
    "kendall1938": "M. G. Kendall, “A new measure of rank correlation,” *Biometrika*, vol. 30, no. 1–2, pp. 81–93, 1938.",
    "lundberg2017": "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Proc. "
                    "Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 4765–4774.",
    "warnecke2020": "A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck, “Evaluating explanation methods for deep "
                    "learning in security,” in *Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P)*, 2020, pp. 158–174.",
    "geirhos2020": "R. Geirhos et al., “Shortcut learning in deep neural networks,” *Nat. Mach. Intell.*, vol. 2, "
                   "no. 11, pp. 665–673, 2020.",
    "madry2018": "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant "
                 "to adversarial attacks,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.",
    "athalye2018": "A. Athalye, N. Carlini, and D. Wagner, “Obfuscated gradients give a false sense of security: "
                   "Circumventing defenses to adversarial examples,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2018, "
                   "pp. 274–283.",
    "apruzzese2022": "G. Apruzzese, M. Andreolini, L. Ferretti, M. Marchetti, and M. Colajanni, “Modeling realistic "
                     "adversarial attacks against network intrusion detection systems,” *Digit. Threats: Res. "
                     "Pract.*, vol. 3, no. 3, Art. no. 31, 2022.",
}
