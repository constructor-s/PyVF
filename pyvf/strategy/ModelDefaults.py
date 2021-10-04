import numpy as np
from ..strategy import PATTERN_P24D2

nan = np.nan

R_P24D2_PARAMETERS = {
    "eval_pattern": PATTERN_P24D2,
    "gh_percentile": (46-1)*1.0/(52-1),
    "intercept": [29.9288847968192, 30.2971485473839, 30.3855714453976, 30.1941534908604, 31.5131875561149, 32.1612921592305, 32.5295559097952, 32.6179788078089, 32.4265608532717, 31.9553020461835, 32.2222672714207, 33.1502127270873, 33.7983173302029, 34.1665810807676, 34.2550039787813, 34.0635860242441, 33.5923272171559, 32.8412275575167, 32.0561239427366, 33.2639102509542, 34.1918557066208, 34.8399603097364, 35.2082240603011, 35.2966469583148, 35.1052290037776, nan, 33.8828705370502, 32.5023847308313, 33.7101710390488, 34.6381164947154, 35.286221097831, 35.6544848483957, 35.7429077464094, 35.5514897918722, nan, 34.3291313251449, 33.5610496357045, 34.4889950913711, 35.1370996944868, 35.5053634450514, 35.5937863430652, 35.4023683885279, 34.9311095814397, 34.1800099218006, 33.7444914965879, 34.3925960997036, 34.7608598502683, 34.849282748282, 34.6578647937447, 34.1866059866565, 33.0527103134815, 33.4209740640462, 33.5093969620599, 33.3179790075226],
    "slope": [-0.063158074516158, -0.0601260960002938, -0.060847146077044, -0.0653212247464087, -0.0634369015893648, -0.0566518944808862, -0.053619915965022, -0.0543409660417722, -0.0588150447111369, -0.067042151973116, -0.0699579364536407, -0.0594199007525477, -0.0526348936440691, -0.0496029151282049, -0.0503239652049551, -0.0547980438743197, -0.0630251511362988, -0.0750052869908923, -0.0827211791089858, -0.0684301148152783, -0.0578920791141852, -0.0511070720057066, -0.0480750934898424, -0.0487961435665926, -0.0532702222359573, nan, -0.0734774653525299, -0.083682536669078, -0.0693914723753705, -0.0588534366742774, -0.0520684295657988, -0.0490364510499346, -0.0497575011266848, -0.0542315797960495, nan, -0.0744388229126221, -0.0728420091339173, -0.0623039734328243, -0.0555189663243457, -0.0524869878084815, -0.0532080378852317, -0.0576821165545963, -0.0659092238165754, -0.0778893596711689, -0.0682436893898258, -0.0614586822813472, -0.058426703765483, -0.0591477538422332, -0.0636218325115979, -0.071848939773577, -0.0698875774368034, -0.0668555989209392, -0.0675766489976894, -0.0720507276670541],
    "sds_sens": [3.11885761844793, 3.02288993840202, 3.05670553855359, 3.22030441890265, 2.61292545155295, 2.38717449130954, 2.29120681126362, 2.32502241141519, 2.48862129176426, 2.78200345231081, 2.45747749133192, 2.10194325089102, 1.87619229064761, 1.7802246106017, 1.81404021075327, 1.97763909110233, 2.27102125164888, 2.69418669239292, 2.65251373778484, 2.16719621714646, 1.81166197670556, 1.58591101646216, 1.48994333641624, 1.52375893656781, 1.68735781691687, nan, 2.40390541820746, 2.58293339007585, 2.09761586943747, 1.74208162899657, 1.51633066875317, 1.42036298870725, 1.45417858885882, 1.61777746920788, nan, 2.33432507049847, 2.24873644820495, 1.89320220776405, 1.66745124752065, 1.57148356747473, 1.6052991676263, 1.76889804797536, 2.06228020852191, 2.48544564926595, 2.265023713008, 2.03927275276459, 1.94330507271868, 1.97712067287025, 2.14071955321931, 2.43410171376586, 2.63179518448501, 2.53582750443909, 2.56964310459066, 2.73324198493973],
    "sds_td": [3.06462364730015, 2.97317868125532, 3.00681052830772, 3.16551918845736, 2.54329852404934, 2.32677674490727, 2.23533177886244, 2.26896362591484, 2.42767228606447, 2.71145775931134, 2.36868099965171, 2.02708240741241, 1.81056062827034, 1.71911566222551, 1.75274750927791, 1.91145616942755, 2.19524164267441, 2.60410392901852, 2.54077107410727, 2.07409566877074, 1.73249707653143, 1.51597529738937, 1.42453033134454, 1.45816217839694, 1.61687083854657, nan, 2.30951859813754, 2.46781652898225, 2.00114112364572, 1.65954253140642, 1.44302075226435, 1.35157578621952, 1.38520763327192, 1.54391629342155, nan, 2.23656405301253, 2.14981736427665, 1.80821877203735, 1.59169699289529, 1.50025202685045, 1.53388387390286, 1.69259253405249, 1.97637800729936, 2.38524029364346, 2.17852579842425, 1.96200401928218, 1.87055905323735, 1.90419090028975, 2.06289956043938, 2.34668503368625, 2.55394183142503, 2.46249686538019, 2.4961287124326, 2.65483737258223],
    "sds_pd": [2.40137801230585, 2.32253729782898, 2.36069849445417, 2.51586160218142, 1.9909471997875, 1.79510457420857, 1.7162638597317, 1.75442505635689, 1.90958816408414, 2.18175318291346, 1.88656507160233, 1.57372053492134, 1.37787790934241, 1.29903719486554, 1.33719839149073, 1.49236149921798, 1.76452651804729, 2.15369344797866, 2.08823162775033, 1.65838517996727, 1.34554064328628, 1.14969801770735, 1.07085730323048, 1.10901849985567, 1.26418160758293, nan, 1.92551355634361, 2.04909850934638, 1.61925206156333, 1.30640752488234, 1.11056489930341, 1.03172418482654, 1.06988538145173, 1.22504848917898, nan, 1.88638043793967, 1.7691657163905, 1.45632117970951, 1.26047855413058, 1.18163783965371, 1.2197990362789, 1.37496214400615, 1.64712716283547, 2.03629409276684, 1.79528160776779, 1.59943898218886, 1.52059826771199, 1.55875946433718, 1.71392257206443, 1.98608759089375, 2.12744618347825, 2.04860546900139, 2.08676666562658, 2.24192977335383],
    "sds_pdghr": [2.50507906740628, 2.42018806101586, 2.45587136230238, 2.61212897126584, 2.04884370119188, 1.84337838712451, 1.75848738073409, 1.79417068202061, 1.95042829098407, 2.22726020762448, 1.91584171075864, 1.58980208901432, 1.38433677494695, 1.29944576855653, 1.33512906984305, 1.49138667880651, 1.76821859544692, 2.16562481976427, 2.10607309610654, 1.65945916668528, 1.33341954494096, 1.12795423087359, 1.04306322448317, 1.07874652576969, 1.23500413473315, nan, 1.90924227569091, 2.05234962013738, 1.60573569071612, 1.27969606897181, 1.07423075490444, 0.989339748514012, 1.02502304980053, 1.181280658764, nan, 1.85551879972176, 1.75467128285117, 1.42863166110686, 1.22316634703949, 1.13827534064906, 1.17395864193558, 1.33021625089904, 1.60704816753945, 2.00445439185681, 1.78022632134611, 1.57476100727874, 1.48987000088831, 1.52555330217483, 1.6818109111383, 1.95864282777871, 2.1290147356222, 2.04412372923177, 2.07980703051829, 2.23606463948175],
    "pd_thresholds": {
        0.005: [-9.57838474792049, -8.6080281980278, -9.8860017785131, -11.2732151556558, -5.88820596143396, -5.38563320588509, -6.29899479269243, -9.34487573617788, -7.59679100501992, -6.84622170538133, -7.09370423837129, -4.25714801949919, -4.71609462898786, -4.88349376018672, -5.70448583524067, -11.225268121472, -6.47836819603301, -5.78521857131087, -9.04292685655495, -5.71900352601894, -6.12295319634454, -4.63539247321118, -5.25479914490982, -5.88961997900418, -5.69869276223375, nan, -7.91242157915274, -10.4412115571769, -5.59978758968107, -4.12573022909556, -4.13556071753306, -4.87424332687134, -5.33443626283254, -5.70057576117031, nan, -8.49360287879447, -8.23901176154619, -4.37183308534794, -4.37672187208043, -5.81170444482074, -5.98189396877673, -5.81883416098774, -7.28119023726267, -6.76787212147493, -7.6606888322119, -4.89635927653439, -6.59186667229485, -5.8773394058478, -5.05293512718664, -6.96335278118329, -7.87361687812263, -9.12782685709219, -7.53085503873542, -5.67775456304852],
        0.01: [-9.32579950263189, -7.92243841105477, -9.73294180977177, -11.042268133917, -5.88517587519691, -4.79709198539529, -6.29899479269243, -8.85521052104465, -6.65835837596871, -5.99307478997912, -7.09370423837129, -3.96876279190982, -4.14489531086027, -4.88349376018672, -5.3027619733558, -10.5560365124211, -6.20992562930209, -5.6310632684585, -7.84381745211878, -5.38801032033539, -5.34687206047494, -4.63539247321118, -4.46066532396747, -4.67889114671711, -5.26653266911547, nan, -6.13069031493499, -10.123167693267, -5.14005182354254, -3.90369981463212, -4.00859008673879, -4.46926271010171, -5.26487819404068, -5.6486780301069, nan, -6.1465214494429, -6.96206964389179, -4.22432682104926, -4.19117227623977, -5.4310811326749, -5.74398911621523, -5.5079007798632, -7.28119023726267, -6.37239963065268, -6.36097599643678, -4.8010018768088, -6.5766813575809, -5.65482901822933, -4.73604385852852, -6.46439967343203, -7.48918814879862, -7.75512497940167, -7.53085503873542, -5.25787829654028],
        0.05: [-5.81353797920338, -7.43883313270368, -6.87481214961391, -7.56598569151705, -4.86837948321298, -3.75987497941186, -4.67773925852362, -5.28893816541829, -4.15885128709667, -5.01043343540714, -4.99855352449063, -3.3138699058021, -3.09297376896079, -4.41732602018546, -4.54742791547175, -5.59343555747938, -4.70807956918122, -4.27106938789945, -6.37288059775673, -4.22833904587006, -3.8990202927239, -3.17700739815609, -3.30480746923395, -3.62154251857368, -4.51103604777348, nan, -4.43634686154038, -6.42021350353015, -4.1471880615545, -3.35833706566157, -3.23149570637834, -3.43084561432077, -3.44609550540291, -4.54472903048044, nan, -5.13461869426105, -5.44212237253118, -3.34567018209996, -3.12098605733572, -4.45977885024233, -4.33924771167016, -4.74992204282955, -5.16573800346725, -4.79594279307832, -5.13892575309669, -4.00794320452589, -5.04540688646076, -4.63367045199538, -3.94977045114954, -5.06485145126398, -6.47432935221449, -5.76327119051289, -4.7457877895751, -4.28249685547189],
        0.95: [2.42692748726664, 1.46568753423451, 0.597405532268744, 1.51297584909747, 1.82626011042297, 0.611498140168487, 0.0641881153989528, -0.239947800245347, 0.0943892025090385, 1.37007254084029, 1.39230949441222, 0.513720568412257, 0.374127469963615, 0.14280507151338, -0.0685457909137862, -0.321519092001929, 0.425898036545964, 1.21065570470967, 1.10253416069933, 0.0280138313133449, 0.403545892301329, 0.324908811074864, 0.466339924879252, 0.329213200946341, 0.157650264240497, nan, 0.991626421609649, 1.22829649171492, 0.258805613821351, 0.62759546829875, 0.275873934316316, 0.0761479898019356, 0.31371287711614, 0.0, nan, 0.883888477888484, 0.284328984447572, 0.253925416331971, 0.388991948285218, -0.128101314151107, -0.244518704108974, -0.0018028502874145, 0.694142082833819, 1.86887086869607, 0.975374483181097, 0.208508052640447, -0.0277355149319497, -0.0682297408859622, 0.377246421248531, 1.71602231589052, 1.8655303953287, 0.681777217412314, 1.34797985739702, 2.30864289594016]
    },
    "td_thresholds": {
        0.005: [-9.07624225133356, -8.65230460784607, -9.79558982685199, -11.3402012308222, -6.58009965757423, -4.71957543232712, -6.16651478046901, -9.41186181134433, -4.95652802648064, -5.73164647187717, -7.61973463213566, -3.46617888042931, -3.25921006081303, -4.60659808156761, -4.6820024492295, -8.3784818744024, -5.69031470536515, -5.20337385782603, -7.65463536857857, -4.95281301759168, -4.3463864351175, -3.60794907609554, -4.63869466627851, -4.57967768452695, -4.40028504726675, nan, -5.45144460163916, -8.80620205648499, -4.17120594790492, -3.12962541238426, -3.02569603842072, -3.70052903714767, -4.1195557463835, -5.13490026472108, nan, -7.76346381923025, -7.61811168128558, -3.34922408239563, -3.32117772251594, -4.28738622251344, -4.83803380961754, -4.89517055391366, -5.89021506924548, -5.98712526699749, -6.24313774443502, -5.5206991159787, -6.02155080927734, -5.2599427577964, -3.74482521153062, -5.34691775375437, -7.64979293496288, -8.05350776553437, -8.02586311996259, -5.82849816658851],
        0.01: [-9.07624225133356, -8.56933059536568, -9.73653567663218, -10.5177054341084, -6.26545262569099, -4.41616962706587, -5.72254187627861, -8.69078761450664, -4.77400233679939, -5.73164647187717, -7.12915681457489, -3.42933854196273, -3.11393775435539, -4.19040617307531, -4.6820024492295, -8.3784818744024, -4.54922211041042, -4.20140050426013, -6.42281164541471, -4.95281301759168, -3.44336775129022, -3.4668935573598, -3.94869466627852, -3.56150115302355, -4.18347524276641, nan, -5.40074515054591, -8.18269218797346, -4.05981982906851, -2.97507859610838, -2.95384160561993, -3.56518843224985, -3.56041797434668, -5.13490026472108, nan, -6.7139243045623, -6.46210899633032, -2.835983426721, -2.86148068135037, -4.25117020092559, -4.29488799418079, -4.33186657136484, -5.80387398604577, -4.76689957504493, -5.41977155005334, -5.5206991159787, -6.02155080927734, -5.18212201006351, -3.55552903339689, -4.80970581729843, -7.34648084888716, -7.2438362434659, -7.52237467119621, -5.77878316449825],
        0.05: [-5.43834569872036, -6.42693477136007, -6.61304838862085, -6.77190657552849, -3.2144686734679, -2.48732273973269, -4.12923864940043, -5.06189336755218, -3.30362621902098, -4.39643571201158, -4.76594671932378, -2.16415121787066, -2.42968413698249, -2.94953912512659, -3.37064179536868, -4.27567767150234, -3.62174269556904, -3.40841509635499, -5.00185989917761, -3.08967324722222, -2.54465472242713, -1.82464306139974, -2.17949317044104, -2.41767448788586, -3.53612411396844, nan, -3.54770008125097, -5.31406745734844, -2.85276797277288, -1.80721534833639, -1.64143102822514, -2.65835768924472, -2.50369442306226, -3.51148132168094, nan, -4.24917407229045, -4.97200306026774, -1.87787377232716, -1.80596171502602, -3.14094752652778, -3.40130406995125, -3.5318983677145, -3.76564663623215, -3.5845377012016, -3.36324663776111, -2.76038797687596, -4.30199898735166, -4.06382650237905, -2.20433651793288, -3.6542376240911, -5.23684858543203, -5.209071332027, -3.31964472420315, -2.93072378843722],
        0.95: [4.92375774866643, 4.25029111663345, 3.46551513933903, 4.21646208102267, 4.56176457103432, 2.97232272780858, 2.57503722457959, 2.41968119392615, 3.04338254477474, 4.19010133727501, 4.09453589905601, 2.71208082182833, 2.72295705458531, 2.49490822543751, 1.81475789872095, 2.10106531020423, 2.56103024272621, 3.65908966193681, 3.47884920687324, 2.69135038536888, 2.64485616997428, 2.86530241067729, 2.76281671737115, 2.67503818489067, 2.50501188578432, nan, 3.4634204891204, 3.76919603531775, 2.71006575126229, 3.07010759688051, 2.46270583090647, 2.57701727579497, 3.08780906995229, 2.90544499750325, nan, 3.38536593819499, 3.1736809580001, 2.53584159238933, 2.6775801769252, 1.95325432836361, 1.96019947542596, 2.32389634089901, 3.07253667051349, 4.38430655492992, 3.03020122661513, 2.27826375367882, 2.09013453147575, 2.00083306678115, 2.51686464858401, 3.76508570089019, 4.00076917785311, 3.51013515250506, 3.80185883882689, 4.86967338450524]
    },
    "vfi_td2pdcutoff": -20,  # In R visualFields vfindex td2pdcutoff = -20
    "locr_pd": 7,
    "vfi_perc": 0.05,
    # "vfi_region": [5, 4, 4, 5, 4, 4, 3, 3, 4, 4, 5, 4, 3, 2, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 2, 0, 4, 5, 4, 3, 2, 1, 1, 2, 0, 4, 5, 4, 3, 2, 2, 3, 4, 5, 4, 4, 3, 3, 4, 4, 5, 4, 4, 5],
    # "vfi_weight": [0.00, 3.29, 1.28, 0.79, 0.57, 0.45]
    "vfi_weights": [0.45, 0.57, 0.57, 0.45, 0.57, 0.57, 0.79, 0.79, 0.57, 0.57, 0.45, 0.57, 0.79, 1.28, 1.28, 0.79, 0.57, 0.45, 0.45, 0.57, 0.79, 1.28, 3.29, 3.29, 1.28, 0.00, 0.57, 0.45, 0.57, 0.79, 1.28, 3.29, 3.29, 1.28, 0.00, 0.57, 0.45, 0.57, 0.79, 1.28, 1.28, 0.79, 0.57, 0.45, 0.57, 0.57, 0.79, 0.79, 0.57, 0.57, 0.45, 0.57, 0.57, 0.45]
}
# SDS seems to stand for standard deviation squared
# The following snippet is how R visualFields calculated the weights
# which is translated below to Python
#     texteval <- paste("vfenv$nv$", td$tpattern[i], "_",
#       td$talgorithm[i], "$sds", sep = "")
#     wgt <- 1/eval(parse(text = texteval))
R_P24D2_PARAMETERS["md_weights"] = 1.0 / np.array(R_P24D2_PARAMETERS["sds_td"])
# Set all nans to zero since they do not contribute towards MD
R_P24D2_PARAMETERS["md_weights"][np.isnan(R_P24D2_PARAMETERS["md_weights"])] = 0.0
R_P24D2_PARAMETERS["md_weights"] /= R_P24D2_PARAMETERS["md_weights"].sum()

R_P24D2_PARAMETERS["psd_weights"] = 1.0 / np.array(R_P24D2_PARAMETERS["sds_pd"])
R_P24D2_PARAMETERS["psd_weights"][np.isnan(R_P24D2_PARAMETERS["psd_weights"])] = 0.0
R_P24D2_PARAMETERS["psd_weights"] /= R_P24D2_PARAMETERS["psd_weights"].sum()

SITAS_P24D2_PARAMETERS = dict(R_P24D2_PARAMETERS)
SITAS_P24D2_PARAMETERS.update({
    # Fitted parameters
    "md_threshold": -11.,  # -19.6,
    # "gh_percentile": 0.85,
    "gh_995": +4.3,  # +3.,
    "gh_005": -2.3,  # -3,
    "delta_030": 14.,  # 15.,  # 9.101009845733643,
    "delta_010": 21.,  # 27.,  # 28.43162441253662,
    "sigma_005": 27.,  # 15.,  # 33.626792907714844,
    'td_thresholds': {0.005: [-11.022124767303467, -10.230344295501709, -11.051375389099121, -13.070459365844727, -8.70739459991455, -7.291216611862183, -6.95082426071167, -7.186275959014893, -8.050535202026367, -10.227993965148926, -9.247145652770996, -6.6933958530426025, -5.789477825164795, -5.403820037841797, -5.825235366821289, -6.288671970367432, -7.978667259216309, -10.285540580749512, -14.707125663757324, -7.999345779418945, -5.578659296035767, -4.9598212242126465, -4.98465895652771, -4.995375394821167, -5.744681358337402, nan, -8.677726745605469, -14.243969440460205, -7.8030171394348145, -5.417463064193726, -4.79070520401001, -4.761359453201294, -5.108295440673828, -5.602008104324341, nan, -7.830402851104736, -8.243478298187256, -5.7039477825164795, -4.811268091201782, -4.7844016551971436, -5.22209620475769, -5.831455945968628, -6.769883155822754, -7.716282367706299, -7.105993270874023, -5.889773845672607, -5.7196877002716064, -5.735462427139282, -6.485659837722778, -6.774456977844238, -8.265504360198975, -7.443937063217163, -7.385268211364746, -7.468004941940308], 0.01: [-8.14381456375122, -7.702736139297485, -8.33258867263794, -9.741442203521729, -6.5256288051605225, -5.7966673374176025, -5.556833028793335, -5.507088661193848, -6.218804836273193, -7.746746063232422, -6.7090630531311035, -5.242440223693848, -4.613900899887085, -4.663769483566284, -4.730679273605347, -4.978140115737915, -5.8601319789886475, -7.442516803741455, -9.741243839263916, -5.891556739807129, -4.387773275375366, -4.05527400970459, -4.129007339477539, -4.1391448974609375, -4.573766708374023, nan, -6.1216936111450195, -9.895084381103516, -5.5669145584106445, -4.246459007263184, -3.8343390226364136, -3.7904168367385864, -4.230675935745239, -4.63176703453064, nan, -5.860702991485596, -6.29090690612793, -4.438966512680054, -3.894078850746155, -3.9583030939102173, -4.303319454193115, -4.90677547454834, -5.421975135803223, -5.861889123916626, -5.309291124343872, -4.328722238540649, -4.321670770645142, -4.707412004470825, -5.266083002090454, -5.640304803848267, -6.270390510559082, -5.5907886028289795, -5.641513824462891, -5.907605171203613], 0.02: [-6.082267999649048, -6.1908204555511475, -6.579586744308472, -7.130999803543091, -4.683220863342285, -4.429713487625122, -4.39408540725708, -4.619420528411865, -4.771772623062134, -5.401000022888184, -4.803593635559082, -3.826294183731079, -3.7088191509246826, -3.685380458831787, -3.8229514360427856, -3.862686038017273, -4.368636608123779, -5.031782627105713, -7.514925003051758, -4.364547491073608, -3.4253289699554443, -3.270645260810852, -3.3411279916763306, -3.360056161880493, -3.7435314655303955, nan, -4.5975682735443115, -7.739726781845093, -4.468634605407715, -3.3249601125717163, -3.1104239225387573, -3.2045881748199463, -3.517402410507202, -3.6557823419570923, nan, -4.48322606086731, -4.969212770462036, -3.4092379808425903, -3.1112754344940186, -3.2053093910217285, -3.5609973669052124, -4.0422043800354, -4.325517416000366, -4.778360605239868, -4.588263034820557, -3.741696834564209, -3.614999532699585, -3.7592856884002686, -4.120122194290161, -4.368512868881226, -4.837908983230591, -4.387370586395264, -4.501818418502808, -4.682923316955566], 0.05: [-4.257514953613281, -4.095043420791626, -4.383092880249023, -4.795202255249023, -3.26286780834198, -3.169822335243225, -3.147794008255005, -3.140360713005066, -3.455276131629944, -3.869042992591858, -3.2911406755447388, -2.6809715032577515, -2.7600388526916504, -2.6593352556228638, -2.8117313385009766, -3.0342365503311157, -3.2836796045303345, -3.4434067010879517, -4.638925313949585, -2.947088599205017, -2.4699442386627197, -2.4263256788253784, -2.582234501838684, -2.584608793258667, -2.8750680685043335, nan, -3.1373443603515625, -4.807973146438599, -2.994279384613037, -2.5904369354248047, -2.4416356086730957, -2.411930561065674, -2.6462072134017944, -2.7136019468307495, nan, -3.033340334892273, -3.3950358629226685, -2.809414863586426, -2.3767658472061157, -2.405144453048706, -2.6841859817504883, -2.7130998373031616, -2.837856888771057, -3.1856080293655396, -3.234857678413391, -2.805230736732483, -2.613497734069824, -2.705932378768921, -2.9556567668914795, -2.9100866317749023, -3.319515347480774, -3.0366783142089844, -3.0918952226638794, -3.0375514030456543]},
    'pd_thresholds': {0.005: [-10.687548160552979, -10.551031589508057, -11.358283042907715, -13.409326076507568, -8.253836631774902, -7.329394340515137, -6.9973084926605225, -7.34782075881958, -8.34098768234253, -10.301810264587402, -8.99294137954712, -6.799151182174683, -5.726437330245972, -5.5655858516693115, -5.967278957366943, -6.436421632766724, -7.8825037479400635, -10.571892261505127, -14.80471420288086, -7.95052433013916, -5.997351884841919, -5.242029666900635, -5.232167482376099, -5.553643226623535, -5.761960506439209, nan, -8.55177116394043, -14.35744571685791, -7.945353269577026, -5.9327392578125, -5.123645067214966, -5.248075008392334, -5.170133829116821, -5.6564249992370605, nan, -8.19634199142456, -8.700233459472656, -6.091200351715088, -5.531444072723389, -5.528722524642944, -5.736550331115723, -6.156968355178833, -6.532009124755859, -7.609784364700317, -7.61454963684082, -6.373286008834839, -6.210064649581909, -6.417257070541382, -6.5023627281188965, -6.838041543960571, -8.691061019897461, -8.188770294189453, -7.636074542999268, -7.559841156005859], 0.01: [-8.305744647979736, -7.991520166397095, -8.573508262634277, -10.159492492675781, -6.376082420349121, -5.6912195682525635, -5.516073942184448, -5.810211181640625, -6.678386449813843, -8.235842227935791, -7.051263093948364, -5.46065616607666, -4.960296154022217, -4.6403913497924805, -4.715347051620483, -5.155905485153198, -6.198256015777588, -7.40047812461853, -9.745672225952148, -6.1005859375, -4.713415145874023, -4.439836025238037, -4.609811782836914, -4.712674140930176, -4.879121780395508, nan, -6.649466276168823, -9.870686054229736, -6.011683225631714, -4.79951286315918, -4.412220478057861, -4.5858681201934814, -4.501827001571655, -4.842217922210693, nan, -6.2828052043914795, -6.742357015609741, -5.2891199588775635, -4.659421920776367, -4.65042519569397, -5.032391309738159, -5.162537097930908, -5.599059343338013, -6.188753843307495, -6.090777635574341, -5.20139741897583, -5.161804914474487, -5.2354888916015625, -5.3458733558654785, -5.920217752456665, -6.942458868026733, -6.226484537124634, -6.111476421356201, -6.355292558670044], 0.02: [-6.70049262046814, -6.558206081390381, -7.055308103561401, -8.036720275878906, -5.0447773933410645, -4.788046836853027, -4.814892292022705, -4.890756607055664, -5.657125473022461, -6.582312107086182, -5.2474610805511475, -4.488699674606323, -4.0290045738220215, -4.16342306137085, -4.326521396636963, -4.680710077285767, -5.1185338497161865, -6.188861131668091, -7.533287525177002, -4.976276636123657, -3.9799420833587646, -3.9718209505081177, -3.8646589517593384, -4.02280068397522, -4.505364656448364, nan, -5.418616533279419, -7.8795390129089355, -5.068163871765137, -4.025568246841431, -3.894399881362915, -4.042692184448242, -4.357999086380005, -4.379812240600586, nan, -5.338918685913086, -5.531554937362671, -4.4315197467803955, -4.065415382385254, -4.083635091781616, -4.462479829788208, -4.686802387237549, -4.979473352432251, -5.460770845413208, -5.2323808670043945, -4.351310729980469, -4.456717491149902, -4.506170034408569, -4.756873369216919, -5.041564702987671, -5.35460901260376, -5.134848594665527, -5.084142446517944, -5.207636117935181], 0.05: [-5.051727294921875, -5.003274440765381, -5.07729959487915, -5.491329908370972, -4.092038154602051, -3.9446393251419067, -3.761656403541565, -4.079515695571899, -4.336587429046631, -4.636681079864502, -3.9478023052215576, -3.3769454956054688, -3.2560133934020996, -3.330649971961975, -3.538459897041321, -3.6876038312911987, -4.197229862213135, -4.470217227935791, -5.411766767501831, -3.853468894958496, -3.276115894317627, -3.1226030588150024, -3.1400363445281982, -3.323104977607727, -3.620315909385681, nan, -4.1728174686431885, -5.540339231491089, -3.8369590044021606, -3.3007224798202515, -3.2436667680740356, -3.280546188354492, -3.392168402671814, -3.6078587770462036, nan, -4.310777187347412, -4.195137023925781, -3.348719835281372, -3.4529258012771606, -3.4480345249176025, -3.652701258659363, -3.635787844657898, -3.9036765098571777, -4.279178619384766, -4.122748613357544, -3.5811023712158203, -3.581080436706543, -3.6283401250839233, -3.912785291671753, -3.791823983192444, -4.300605297088623, -4.093430519104004, -3.9247125387191772, -4.090740919113159]},
    "md_weights": [0.00831055063386572, 0.010037699335216272, 0.009769463493458037, 0.008536618647016138, 0.012694906273249509, 0.01745458536699493, 0.018971313125537122, 0.017424960212379985, 0.014189821132331047, 0.011171824509992018, 0.012259432892482705, 0.020336479937630256, 0.02561337634374366, 0.02623680119387165, 0.023432294979192875, 0.019205262876969414, 0.014674617776998647, 0.012314028876724503, 0.007751344890596243, 0.01650713138494808, 0.02527075511329894, 0.03038275427609892, 0.028743815183089286, 0.024776355837873982, 0.02172518798281512, 0.0, 0.01386549020961117, 0.009409453343247266, 0.017857625136796788, 0.026314830870238774, 0.03112298016789338, 0.029182422460461478, 0.025033061047396984, 0.02222136663950757, 0.0, 0.014961320339011114, 0.016536501434819488, 0.02376105326875104, 0.028242456525443195, 0.028213431697656385, 0.025164878055294028, 0.02128863681423284, 0.01743742540724297, 0.0158817507312707, 0.019265469868380767, 0.022941619207149107, 0.023687188520207892, 0.02188858282099993, 0.019015088968272205, 0.016832877724988777, 0.017817478697789938, 0.01872716272087504, 0.01820426403114974, 0.017334201014936507],
    # These are equivalent to s squared in Heijl et al: A package for the statistical analysis of visual fields
    "sds_pd": [0.4517563065782965, 0.39663539729301095, 0.41536722920531566, 0.5022024304505646, 0.31267248206546505, 0.22612405089034635, 0.2089890794990538, 0.22882906618062887, 0.2789468918782304, 0.3828056286561195, 0.3274722032061771, 0.18368861276136367, 0.14034354227731793, 0.14255097725791074, 0.16397148966307185, 0.19250006184580368, 0.24937818942058634, 0.3859032867735142, 0.5354248857218487, 0.23955448873462865, 0.137393774555195, 0.11377227625802525, 0.12483092280618013, 0.147575095188767, 0.1605595656434601, nan, 0.2848557087243253, 0.4923264345693539, 0.22094734985880962, 0.13092026038725832, 0.11465395934059945, 0.13080699172040253, 0.1433040948647546, 0.14690094120250624, nan, 0.24481200747506543, 0.25316441565028414, 0.14944108121723604, 0.12523499122735474, 0.13158456205777086, 0.13793339603908497, 0.1414776360964589, 0.1653112741742085, 0.24768823544897628, 0.2009279531732106, 0.15099187054585644, 0.14006489687289364, 0.14145397781824012, 0.15243247690124065, 0.19184505413378256, 0.2252581321139808, 0.18958836791953174, 0.18539579285031915, 0.20859042874857975],
    "intercept": [31.349933293, 31.312414416, 31.169739532, 30.882612385, 32.588484259, 33.076235356, 33.133126378, 33.243637973, 32.593688003, 32.25872542 , 32.309453205, 33.403797913, 34.361128539, 34.84212894 , 34.622988482, 34.134933308, 33.28108327 , 32.776603045, 31.094173661, 32.699711387, 34.157334348, 35.302164332, 35.768661454, 35.596776643, 34.872284668,          nan, 33.135021322, 31.203513499, 32.744357571, 34.298435927, 35.319608295, 35.892821992, 35.734421575, 35.106355909,          nan, 33.107561327, 32.274466065, 33.745948497, 34.747230416, 35.211243942, 35.009893642, 34.41222315 , 33.63516923 , 32.906396341, 32.488400928, 33.413697678, 33.819462803, 33.748204153, 33.36543537 , 33.127312154, 31.685092576, 32.1794866  , 32.317946534, 32.235124663],
    "slope": [-0.075817247, -0.072909852, -0.075175658, -0.079044156, -0.070353108, -0.065272186, -0.062907624, -0.069014646, -0.067426022, -0.075163357, -0.066401198, -0.057082506, -0.057386475, -0.061507134, -0.062171979, -0.065100995, -0.06471435 , -0.072229495, -0.075731615, -0.058483369, -0.052979535, -0.056006571, -0.059697527, -0.061627668, -0.06043234 ,          nan, -0.06317035 , -0.075984626, -0.056088778, -0.051385679, -0.052167634, -0.056535479, -0.057938387, -0.059073015,          nan, -0.056320839, -0.057522544, -0.051461355, -0.050996777, -0.053066416, -0.051570916, -0.051679014, -0.051249615, -0.053477272, -0.050593126, -0.047800924, -0.046561642, -0.046997395, -0.04659614 , -0.053001291, -0.047593823, -0.046412432, -0.046091855, -0.04601346 ]
})
