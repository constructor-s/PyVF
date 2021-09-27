import numpy as np

nan = np.nan

DEFAULT_PARAMETERS = {
    "md_threshold": -19.6,
    "gh_percentile": 0.85,
    "gh_995": 3,
    "gh_005": -5,  # -3,
    "delta_030": 15,  # 9.101009845733643,
    "delta_010": 27,  # 28.43162441253662,
    "sigma_005": 15,  # 33.626792907714844,
    "td_thresholds": {
        0.005: [-11.37406873703003, -10.655344009399414, -11.051375389099121, -13.213706016540527, -8.873942852020264,
                -7.557756185531616, -7.071288824081421, -7.186275959014893, -8.251474857330322, -10.227993965148926,
                -9.433214664459229, -6.946624517440796, -6.07136344909668, -5.817965745925903, -5.968147039413452,
                -6.611950397491455, -8.013483047485352, -10.285540580749512, -14.707125663757324, -8.070278644561768,
                -6.033536434173584, -5.132990598678589, -4.98465895652771, -5.375471115112305, -5.944798707962036, nan,
                -8.737308502197266, -14.379388809204102, -7.8030171394348145, -5.46037220954895, -4.823517560958862,
                -4.932587385177612, -5.312694549560547, -5.634510278701782, nan, -8.178838729858398, -8.243478298187256,
                -5.821672677993774, -5.114696025848389, -5.051966428756714, -5.500741720199585, -5.839261293411255,
                -6.830350160598755, -7.759766340255737, -7.2539732456207275, -6.148119211196899, -5.7196877002716064,
                -6.138591527938843, -6.595094442367554, -6.774456977844238, -8.265504360198975, -7.529709339141846,
                -7.385268211364746, -7.632091045379639],
        0.01: [-8.14381456375122, -7.979582071304321, -8.33258867263794, -9.688745498657227, -6.470861434936523,
               -5.7966673374176025, -5.556833028793335, -5.731029987335205, -6.218804836273193, -7.746746063232422,
               -6.7090630531311035, -5.242440223693848, -4.613900899887085, -4.663769483566284, -4.779531955718994,
               -4.978140115737915, -5.8601319789886475, -7.442516803741455, -9.741243839263916, -5.891556739807129,
               -4.558147192001343, -4.05527400970459, -4.159295320510864, -4.1391448974609375, -4.618199586868286, nan,
               -6.2191526889801025, -9.895084381103516, -5.5669145584106445, -4.291266679763794, -3.9719483852386475,
               -3.7904168367385864, -3.999794840812683, -4.63176703453064, nan, -5.860702991485596, -5.983623027801514,
               -4.438966512680054, -4.0029990673065186, -4.045928478240967, -4.303319454193115, -4.90677547454834,
               -5.421975135803223, -5.861889123916626, -5.734838485717773, -4.328722238540649, -4.583731412887573,
               -4.732154846191406, -5.193987131118774, -5.55797815322876, -6.270390510559082, -5.858694553375244,
               -5.781335115432739, -5.873093843460083],
        0.02: [-6.082267999649048, -6.1908204555511475, -6.579586744308472, -7.130999803543091, -4.581712007522583,
               -4.429713487625122, -4.212713956832886, -4.654659986495972, -4.771772623062134, -5.401000022888184,
               -4.699274063110352, -3.826294183731079, -3.7088191509246826, -3.532896637916565, -3.7939435243606567,
               -3.862686038017273, -4.368636608123779, -5.031782627105713, -7.514925003051758, -4.315340042114258,
               -3.245321750640869, -3.270645260810852, -3.3411279916763306, -3.360056161880493, -3.7435314655303955,
               nan,
               -4.561853647232056, -7.739726781845093, -4.468634605407715, -3.260526657104492, -3.031688094139099,
               -3.2045881748199463, -3.517402410507202, -3.548957586288452, nan, -4.48322606086731, -4.790985822677612,
               -3.4092379808425903, -2.9361201524734497, -3.0807807445526123, -3.5609973669052124, -4.0422043800354,
               -4.128525257110596, -4.62067437171936, -4.642215251922607, -3.6141536235809326, -3.4494961500167847,
               -3.679123044013977, -3.8788576126098633, -4.360651969909668, -4.837908983230591, -4.363459587097168,
               -4.501818418502808, -4.489861965179443],
        0.05: [-3.9385751485824585, -3.945170283317566, -4.1378443241119385, -4.564561605453491, -3.08049213886261,
               -3.0061100721359253, -2.87089741230011, -2.9597562551498413, -3.381892204284668, -3.7651209831237793,
               -3.1087392568588257, -2.480772376060486, -2.4482256174087524, -2.406374216079712, -2.542291045188904,
               -2.726156711578369, -3.0942790508270264, -3.3690911531448364, -4.4732444286346436, -2.9405529499053955,
               -2.2676594257354736, -2.3122764825820923, -2.1724830865859985, -2.3977667093276978, -2.4514254331588745,
               nan,
               -2.9475293159484863, -4.574005126953125, -2.9519726037979126, -2.3474708795547485, -2.0931038856506348,
               -2.1119489669799805, -2.2932716608047485, -2.380431652069092, nan, -2.874863386154175,
               -3.1295605897903442,
               -2.5957809686660767, -2.14504873752594, -2.163594961166382, -2.612496256828308, -2.5733319520950317,
               -2.7297451496124268, -2.9368526935577393, -2.9976483583450317, -2.494205355644226, -2.4999141693115234,
               -2.505223035812378, -2.771377205848694, -2.763065814971924, -3.115523338317871, -2.892114043235779,
               -2.916273355484009, -3.0375514030456543]},
    "pd_thresholds": {
        0.005: [-10.601900100708008, -10.6086745262146, -10.911179065704346, -13.196418762207031, -8.041792392730713,
                -7.182874917984009, -6.898180961608887, -7.146293640136719, -8.185040473937988, -10.174447059631348,
                -8.815481662750244, -6.494378089904785, -5.6896984577178955, -5.664662599563599, -5.777994632720947,
                -6.419536352157593, -7.88475227355957, -10.402885437011719, -14.484954833984375, -7.7886738777160645,
                -5.806567430496216, -5.262685060501099, -5.0774617195129395, -5.406658172607422, -5.8829638957977295,
                nan,
                -8.250314235687256, -14.33190393447876, -7.867198467254639, -5.869597911834717, -4.973596572875977,
                -5.004702091217041, -5.214554071426392, -5.451503038406372, nan, -7.884553670883179, -8.223504066467285,
                -5.840505599975586, -5.3024373054504395, -5.249982833862305, -5.582663297653198, -5.9125354290008545,
                -6.626039266586304, -7.428748369216919, -7.37362813949585, -6.190459966659546, -6.086302995681763,
                -6.249268054962158, -6.5937275886535645, -6.7876434326171875, -8.318763732910156, -8.043383121490479,
                -7.705788850784302, -7.499807119369507],
        0.01: [-7.8044373989105225, -7.579102516174316, -8.383288383483887, -9.88568925857544, -6.196738004684448,
               -5.620035648345947, -5.046980857849121, -5.571954011917114, -6.307464122772217, -7.702406167984009,
               -6.478914737701416, -4.959868431091309, -4.769948482513428, -4.525350570678711, -4.883265018463135,
               -4.999692440032959, -5.769596099853516, -7.50554347038269, -9.501469612121582, -5.911007404327393,
               -4.519992828369141, -4.092721939086914, -4.207835674285889, -4.699853897094727, -4.960667133331299, nan,
               -6.478224515914917, -9.611674308776855, -5.790405988693237, -4.273298501968384, -4.199483156204224,
               -4.336620807647705, -4.554642200469971, -4.613354206085205, nan, -5.915425539016724, -6.647728204727173,
               -5.100634813308716, -4.27729606628418, -4.438990592956543, -4.928002119064331, -5.071532964706421,
               -5.42884373664856, -5.903240203857422, -5.892519235610962, -4.955704212188721, -4.953541994094849,
               -4.963058710098267, -5.079463958740234, -5.714881420135498, -6.5637266635894775, -5.95746636390686,
               -5.984622001647949, -5.992669105529785],
        0.02: [-6.035579681396484, -6.056212663650513, -6.443699836730957, -7.740461349487305, -4.892344951629639,
               -4.289147138595581, -4.598344802856445, -4.613813161849976, -5.179100036621094, -6.265207767486572,
               -4.943010568618774, -3.936710000038147, -4.066634654998779, -3.756205916404724, -4.006028652191162,
               -4.406725883483887, -4.919477939605713, -5.857668876647949, -7.122624158859253, -4.5345139503479,
               -3.762368083000183, -3.7534291744232178, -3.471523642539978, -3.7204898595809937, -4.256702423095703,
               nan,
               -5.042004823684692, -7.408905744552612, -4.414096832275391, -3.604548454284668, -3.437385678291321,
               -3.889699697494507, -3.6831456422805786, -4.140583038330078, nan, -4.831751585006714, -5.359666585922241,
               -4.087791442871094, -3.5169782638549805, -3.8743104934692383, -3.7930082082748413, -4.391925811767578,
               -4.507098436355591, -4.987668991088867, -4.567487001419067, -4.407484769821167, -3.969321131706238,
               -4.211819887161255, -4.511439323425293, -4.83476996421814, -5.127641916275024, -4.95258903503418,
               -4.756721019744873, -4.8484601974487305],
        0.05: [-4.535223484039307, -4.498387336730957, -4.544172763824463, -5.087820053100586, -3.383420467376709,
               -3.283785104751587, -3.3149542808532715, -3.503061294555664, -3.5458563566207886, -4.254762649536133,
               -3.3917417526245117, -2.805738091468811, -2.71648108959198, -2.816029906272888, -3.216481566429138,
               -3.198686718940735, -3.725896716117859, -4.075464963912964, -4.822077751159668, -3.2803127765655518,
               -2.766081690788269, -2.7571576833724976, -2.673731803894043, -2.750171422958374, -3.038131356239319, nan,
               -3.6693869829177856, -5.040565013885498, -3.342800498008728, -2.8390140533447266, -2.596175193786621,
               -2.7204370498657227, -2.9388099908828735, -3.118218183517456, nan, -3.6349780559539795,
               -3.726341724395752,
               -2.8266159296035767, -2.6416903734207153, -2.9160135984420776, -2.9661890268325806, -3.232709050178528,
               -3.4087802171707153, -3.5836464166641235, -3.3673242330551147, -3.1230238676071167, -2.9579203128814697,
               -2.9614280462265015, -3.254051446914673, -3.2021106481552124, -3.723511576652527, -3.3738988637924194,
               -3.3606438636779785, -3.520809054374695]}
}

R_PARAMETERS = {
    "gh_percentile": (46-1)*1.0/(52-1),
    "intercept": [29.9288847968192, 30.2971485473839, 30.3855714453976, 30.1941534908604, 31.5131875561149, 32.1612921592305, 32.5295559097952, 32.6179788078089, 32.4265608532717, 31.9553020461835, 32.2222672714207, 33.1502127270873, 33.7983173302029, 34.1665810807676, 34.2550039787813, 34.0635860242441, 33.5923272171559, 32.8412275575167, 32.0561239427366, 33.2639102509542, 34.1918557066208, 34.8399603097364, 35.2082240603011, 35.2966469583148, 35.1052290037776, nan, 33.8828705370502, 32.5023847308313, 33.7101710390488, 34.6381164947154, 35.286221097831, 35.6544848483957, 35.7429077464094, 35.5514897918722, nan, 34.3291313251449, 33.5610496357045, 34.4889950913711, 35.1370996944868, 35.5053634450514, 35.5937863430652, 35.4023683885279, 34.9311095814397, 34.1800099218006, 33.7444914965879, 34.3925960997036, 34.7608598502683, 34.849282748282, 34.6578647937447, 34.1866059866565, 33.0527103134815, 33.4209740640462, 33.5093969620599, 33.3179790075226],
    "slope": [-0.063158074516158, -0.0601260960002938, -0.060847146077044, -0.0653212247464087, -0.0634369015893648, -0.0566518944808862, -0.053619915965022, -0.0543409660417722, -0.0588150447111369, -0.067042151973116, -0.0699579364536407, -0.0594199007525477, -0.0526348936440691, -0.0496029151282049, -0.0503239652049551, -0.0547980438743197, -0.0630251511362988, -0.0750052869908923, -0.0827211791089858, -0.0684301148152783, -0.0578920791141852, -0.0511070720057066, -0.0480750934898424, -0.0487961435665926, -0.0532702222359573, nan, -0.0734774653525299, -0.083682536669078, -0.0693914723753705, -0.0588534366742774, -0.0520684295657988, -0.0490364510499346, -0.0497575011266848, -0.0542315797960495, nan, -0.0744388229126221, -0.0728420091339173, -0.0623039734328243, -0.0555189663243457, -0.0524869878084815, -0.0532080378852317, -0.0576821165545963, -0.0659092238165754, -0.0778893596711689, -0.0682436893898258, -0.0614586822813472, -0.058426703765483, -0.0591477538422332, -0.0636218325115979, -0.071848939773577, -0.0698875774368034, -0.0668555989209392, -0.0675766489976894, -0.0720507276670541],
    "sds_sens": [3.11885761844793, 3.02288993840202, 3.05670553855359, 3.22030441890265, 2.61292545155295, 2.38717449130954, 2.29120681126362, 2.32502241141519, 2.48862129176426, 2.78200345231081, 2.45747749133192, 2.10194325089102, 1.87619229064761, 1.7802246106017, 1.81404021075327, 1.97763909110233, 2.27102125164888, 2.69418669239292, 2.65251373778484, 2.16719621714646, 1.81166197670556, 1.58591101646216, 1.48994333641624, 1.52375893656781, 1.68735781691687, nan, 2.40390541820746, 2.58293339007585, 2.09761586943747, 1.74208162899657, 1.51633066875317, 1.42036298870725, 1.45417858885882, 1.61777746920788, nan, 2.33432507049847, 2.24873644820495, 1.89320220776405, 1.66745124752065, 1.57148356747473, 1.6052991676263, 1.76889804797536, 2.06228020852191, 2.48544564926595, 2.265023713008, 2.03927275276459, 1.94330507271868, 1.97712067287025, 2.14071955321931, 2.43410171376586, 2.63179518448501, 2.53582750443909, 2.56964310459066, 2.73324198493973],
    "sds_td": [3.06462364730015, 2.97317868125532, 3.00681052830772, 3.16551918845736, 2.54329852404934, 2.32677674490727, 2.23533177886244, 2.26896362591484, 2.42767228606447, 2.71145775931134, 2.36868099965171, 2.02708240741241, 1.81056062827034, 1.71911566222551, 1.75274750927791, 1.91145616942755, 2.19524164267441, 2.60410392901852, 2.54077107410727, 2.07409566877074, 1.73249707653143, 1.51597529738937, 1.42453033134454, 1.45816217839694, 1.61687083854657, nan, 2.30951859813754, 2.46781652898225, 2.00114112364572, 1.65954253140642, 1.44302075226435, 1.35157578621952, 1.38520763327192, 1.54391629342155, nan, 2.23656405301253, 2.14981736427665, 1.80821877203735, 1.59169699289529, 1.50025202685045, 1.53388387390286, 1.69259253405249, 1.97637800729936, 2.38524029364346, 2.17852579842425, 1.96200401928218, 1.87055905323735, 1.90419090028975, 2.06289956043938, 2.34668503368625, 2.55394183142503, 2.46249686538019, 2.4961287124326, 2.65483737258223],
    "sds_pd": [2.40137801230585, 2.32253729782898, 2.36069849445417, 2.51586160218142, 1.9909471997875, 1.79510457420857, 1.7162638597317, 1.75442505635689, 1.90958816408414, 2.18175318291346, 1.88656507160233, 1.57372053492134, 1.37787790934241, 1.29903719486554, 1.33719839149073, 1.49236149921798, 1.76452651804729, 2.15369344797866, 2.08823162775033, 1.65838517996727, 1.34554064328628, 1.14969801770735, 1.07085730323048, 1.10901849985567, 1.26418160758293, nan, 1.92551355634361, 2.04909850934638, 1.61925206156333, 1.30640752488234, 1.11056489930341, 1.03172418482654, 1.06988538145173, 1.22504848917898, nan, 1.88638043793967, 1.7691657163905, 1.45632117970951, 1.26047855413058, 1.18163783965371, 1.2197990362789, 1.37496214400615, 1.64712716283547, 2.03629409276684, 1.79528160776779, 1.59943898218886, 1.52059826771199, 1.55875946433718, 1.71392257206443, 1.98608759089375, 2.12744618347825, 2.04860546900139, 2.08676666562658, 2.24192977335383],
    "sds_pdghr": [2.50507906740628, 2.42018806101586, 2.45587136230238, 2.61212897126584, 2.04884370119188, 1.84337838712451, 1.75848738073409, 1.79417068202061, 1.95042829098407, 2.22726020762448, 1.91584171075864, 1.58980208901432, 1.38433677494695, 1.29944576855653, 1.33512906984305, 1.49138667880651, 1.76821859544692, 2.16562481976427, 2.10607309610654, 1.65945916668528, 1.33341954494096, 1.12795423087359, 1.04306322448317, 1.07874652576969, 1.23500413473315, nan, 1.90924227569091, 2.05234962013738, 1.60573569071612, 1.27969606897181, 1.07423075490444, 0.989339748514012, 1.02502304980053, 1.181280658764, nan, 1.85551879972176, 1.75467128285117, 1.42863166110686, 1.22316634703949, 1.13827534064906, 1.17395864193558, 1.33021625089904, 1.60704816753945, 2.00445439185681, 1.78022632134611, 1.57476100727874, 1.48987000088831, 1.52555330217483, 1.6818109111383, 1.95864282777871, 2.1290147356222, 2.04412372923177, 2.07980703051829, 2.23606463948175],
    "td_thresholds": {
        0.005: [-9.57838474792049, -8.6080281980278, -9.8860017785131, -11.2732151556558, -5.88820596143396, -5.38563320588509, -6.29899479269243, -9.34487573617788, -7.59679100501992, -6.84622170538133, -7.09370423837129, -4.25714801949919, -4.71609462898786, -4.88349376018672, -5.70448583524067, -11.225268121472, -6.47836819603301, -5.78521857131087, -9.04292685655495, -5.71900352601894, -6.12295319634454, -4.63539247321118, -5.25479914490982, -5.88961997900418, -5.69869276223375, nan, -7.91242157915274, -10.4412115571769, -5.59978758968107, -4.12573022909556, -4.13556071753306, -4.87424332687134, -5.33443626283254, -5.70057576117031, nan, -8.49360287879447, -8.23901176154619, -4.37183308534794, -4.37672187208043, -5.81170444482074, -5.98189396877673, -5.81883416098774, -7.28119023726267, -6.76787212147493, -7.6606888322119, -4.89635927653439, -6.59186667229485, -5.8773394058478, -5.05293512718664, -6.96335278118329, -7.87361687812263, -9.12782685709219, -7.53085503873542, -5.67775456304852],
        0.01: [-9.32579950263189, -7.92243841105477, -9.73294180977177, -11.042268133917, -5.88517587519691, -4.79709198539529, -6.29899479269243, -8.85521052104465, -6.65835837596871, -5.99307478997912, -7.09370423837129, -3.96876279190982, -4.14489531086027, -4.88349376018672, -5.3027619733558, -10.5560365124211, -6.20992562930209, -5.6310632684585, -7.84381745211878, -5.38801032033539, -5.34687206047494, -4.63539247321118, -4.46066532396747, -4.67889114671711, -5.26653266911547, nan, -6.13069031493499, -10.123167693267, -5.14005182354254, -3.90369981463212, -4.00859008673879, -4.46926271010171, -5.26487819404068, -5.6486780301069, nan, -6.1465214494429, -6.96206964389179, -4.22432682104926, -4.19117227623977, -5.4310811326749, -5.74398911621523, -5.5079007798632, -7.28119023726267, -6.37239963065268, -6.36097599643678, -4.8010018768088, -6.5766813575809, -5.65482901822933, -4.73604385852852, -6.46439967343203, -7.48918814879862, -7.75512497940167, -7.53085503873542, -5.25787829654028],
        0.05: [-5.81353797920338, -7.43883313270368, -6.87481214961391, -7.56598569151705, -4.86837948321298, -3.75987497941186, -4.67773925852362, -5.28893816541829, -4.15885128709667, -5.01043343540714, -4.99855352449063, -3.3138699058021, -3.09297376896079, -4.41732602018546, -4.54742791547175, -5.59343555747938, -4.70807956918122, -4.27106938789945, -6.37288059775673, -4.22833904587006, -3.8990202927239, -3.17700739815609, -3.30480746923395, -3.62154251857368, -4.51103604777348, nan, -4.43634686154038, -6.42021350353015, -4.1471880615545, -3.35833706566157, -3.23149570637834, -3.43084561432077, -3.44609550540291, -4.54472903048044, nan, -5.13461869426105, -5.44212237253118, -3.34567018209996, -3.12098605733572, -4.45977885024233, -4.33924771167016, -4.74992204282955, -5.16573800346725, -4.79594279307832, -5.13892575309669, -4.00794320452589, -5.04540688646076, -4.63367045199538, -3.94977045114954, -5.06485145126398, -6.47432935221449, -5.76327119051289, -4.7457877895751, -4.28249685547189],
        0.95: [2.42692748726664, 1.46568753423451, 0.597405532268744, 1.51297584909747, 1.82626011042297, 0.611498140168487, 0.0641881153989528, -0.239947800245347, 0.0943892025090385, 1.37007254084029, 1.39230949441222, 0.513720568412257, 0.374127469963615, 0.14280507151338, -0.0685457909137862, -0.321519092001929, 0.425898036545964, 1.21065570470967, 1.10253416069933, 0.0280138313133449, 0.403545892301329, 0.324908811074864, 0.466339924879252, 0.329213200946341, 0.157650264240497, nan, 0.991626421609649, 1.22829649171492, 0.258805613821351, 0.62759546829875, 0.275873934316316, 0.0761479898019356, 0.31371287711614, 0.0, nan, 0.883888477888484, 0.284328984447572, 0.253925416331971, 0.388991948285218, -0.128101314151107, -0.244518704108974, -0.0018028502874145, 0.694142082833819, 1.86887086869607, 0.975374483181097, 0.208508052640447, -0.0277355149319497, -0.0682297408859622, 0.377246421248531, 1.71602231589052, 1.8655303953287, 0.681777217412314, 1.34797985739702, 2.30864289594016]
    },
    "pd_thresholds": {
        0.005: [-9.07624225133356, -8.65230460784607, -9.79558982685199, -11.3402012308222, -6.58009965757423, -4.71957543232712, -6.16651478046901, -9.41186181134433, -4.95652802648064, -5.73164647187717, -7.61973463213566, -3.46617888042931, -3.25921006081303, -4.60659808156761, -4.6820024492295, -8.3784818744024, -5.69031470536515, -5.20337385782603, -7.65463536857857, -4.95281301759168, -4.3463864351175, -3.60794907609554, -4.63869466627851, -4.57967768452695, -4.40028504726675, nan, -5.45144460163916, -8.80620205648499, -4.17120594790492, -3.12962541238426, -3.02569603842072, -3.70052903714767, -4.1195557463835, -5.13490026472108, nan, -7.76346381923025, -7.61811168128558, -3.34922408239563, -3.32117772251594, -4.28738622251344, -4.83803380961754, -4.89517055391366, -5.89021506924548, -5.98712526699749, -6.24313774443502, -5.5206991159787, -6.02155080927734, -5.2599427577964, -3.74482521153062, -5.34691775375437, -7.64979293496288, -8.05350776553437, -8.02586311996259, -5.82849816658851],
        0.01: [-9.07624225133356, -8.56933059536568, -9.73653567663218, -10.5177054341084, -6.26545262569099, -4.41616962706587, -5.72254187627861, -8.69078761450664, -4.77400233679939, -5.73164647187717, -7.12915681457489, -3.42933854196273, -3.11393775435539, -4.19040617307531, -4.6820024492295, -8.3784818744024, -4.54922211041042, -4.20140050426013, -6.42281164541471, -4.95281301759168, -3.44336775129022, -3.4668935573598, -3.94869466627852, -3.56150115302355, -4.18347524276641, nan, -5.40074515054591, -8.18269218797346, -4.05981982906851, -2.97507859610838, -2.95384160561993, -3.56518843224985, -3.56041797434668, -5.13490026472108, nan, -6.7139243045623, -6.46210899633032, -2.835983426721, -2.86148068135037, -4.25117020092559, -4.29488799418079, -4.33186657136484, -5.80387398604577, -4.76689957504493, -5.41977155005334, -5.5206991159787, -6.02155080927734, -5.18212201006351, -3.55552903339689, -4.80970581729843, -7.34648084888716, -7.2438362434659, -7.52237467119621, -5.77878316449825],
        0.05: [-5.43834569872036, -6.42693477136007, -6.61304838862085, -6.77190657552849, -3.2144686734679, -2.48732273973269, -4.12923864940043, -5.06189336755218, -3.30362621902098, -4.39643571201158, -4.76594671932378, -2.16415121787066, -2.42968413698249, -2.94953912512659, -3.37064179536868, -4.27567767150234, -3.62174269556904, -3.40841509635499, -5.00185989917761, -3.08967324722222, -2.54465472242713, -1.82464306139974, -2.17949317044104, -2.41767448788586, -3.53612411396844, nan, -3.54770008125097, -5.31406745734844, -2.85276797277288, -1.80721534833639, -1.64143102822514, -2.65835768924472, -2.50369442306226, -3.51148132168094, nan, -4.24917407229045, -4.97200306026774, -1.87787377232716, -1.80596171502602, -3.14094752652778, -3.40130406995125, -3.5318983677145, -3.76564663623215, -3.5845377012016, -3.36324663776111, -2.76038797687596, -4.30199898735166, -4.06382650237905, -2.20433651793288, -3.6542376240911, -5.23684858543203, -5.209071332027, -3.31964472420315, -2.93072378843722],
        0.95: [4.92375774866643, 4.25029111663345, 3.46551513933903, 4.21646208102267, 4.56176457103432, 2.97232272780858, 2.57503722457959, 2.41968119392615, 3.04338254477474, 4.19010133727501, 4.09453589905601, 2.71208082182833, 2.72295705458531, 2.49490822543751, 1.81475789872095, 2.10106531020423, 2.56103024272621, 3.65908966193681, 3.47884920687324, 2.69135038536888, 2.64485616997428, 2.86530241067729, 2.76281671737115, 2.67503818489067, 2.50501188578432, nan, 3.4634204891204, 3.76919603531775, 2.71006575126229, 3.07010759688051, 2.46270583090647, 2.57701727579497, 3.08780906995229, 2.90544499750325, nan, 3.38536593819499, 3.1736809580001, 2.53584159238933, 2.6775801769252, 1.95325432836361, 1.96019947542596, 2.32389634089901, 3.07253667051349, 4.38430655492992, 3.03020122661513, 2.27826375367882, 2.09013453147575, 2.00083306678115, 2.51686464858401, 3.76508570089019, 4.00076917785311, 3.51013515250506, 3.80185883882689, 4.86967338450524]
    }
}
# Not sure why R visualFields used the reciprocal of the standard deviation
# rather than reciprocal of the square of the standard deviation as weights
#     texteval <- paste("vfenv$nv$", td$tpattern[i], "_",
#       td$talgorithm[i], "$sds", sep = "")
#     wgt <- 1/eval(parse(text = texteval))
R_PARAMETERS["md_weights"] = 1.0 / np.array(R_PARAMETERS["sds_td"])
# Set all nans to zero since they do not contribute towards MD
R_PARAMETERS["md_weights"][np.isnan(R_PARAMETERS["md_weights"])] = 0.0
R_PARAMETERS["md_weights"] /= R_PARAMETERS["md_weights"].sum()

R_PARAMETERS["psd_weights"] = 1.0 / np.array(R_PARAMETERS["sds_pd"])
R_PARAMETERS["psd_weights"][np.isnan(R_PARAMETERS["psd_weights"])] = 0.0
R_PARAMETERS["psd_weights"] /= R_PARAMETERS["psd_weights"].sum()
