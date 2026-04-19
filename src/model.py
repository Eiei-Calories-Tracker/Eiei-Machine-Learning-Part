import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

CLASS_NAMES = [
    'BitterMelonSoup', 'BooPadPongali', 'Curried FishCake', 'Dumpling', 'EggsStewed', 
    'Fried Chicken', 'FriedKale', 'Fried Mussel Pancakes', 'Gaeng Jued', 'GaengKeawWan', 
    'GaiYang', 'GoongObWoonSen', 'GoongPao', 'GrilledQquid', 'HoyKraeng', 'HoyLaiPrikPao', 
    'Joke', 'KaiJeowMooSaap', 'KaiThoon', 'KaoManGai', 'KaoMooDang', 'KhanomJeenNamYaKati', 
    'KhaoMokGai', 'KhaoMooTodGratiem', 'KhaoNiewMaMuang', 'KkaoKlukkaphi', 'KorMooYang', 
    'Kuakling', 'KuayJab', 'Kuay TeowReua', 'LarbMoo', 'MassamanGai', 'MooSatay', 'Nam TokMoo', 
    'PadPakBung', 'PadPakRuamMit', 'PadThai', 'PadYordMala', 'PhatKaphrao', 'PorkStickyNoodles', 
    'Roast duck', 'Roast fish', 'Somtam', 'SoninLawEggs', 'Stewed PorkLeg', 'Suki', 
    'TomKhaGai', 'TomYum Goong', 'YamWoonSen', 'Yentafo'
]

def create_model(num_classes=len(CLASS_NAMES), dropout=0.3, weights=GoogLeNet_Weights.IMAGENET1K_V1):
    model = googlenet(weights=weights, aux_logits=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes)
    )
    return model
