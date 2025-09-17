from enum import Enum


class ModelArchs(str, Enum):
    UNET = "unet"
    SEGFORMER = "segformer"


class LabelNames(str, Enum):
    POROSITY = "porosity"
    INCLUSION = "inclusion"
    CRACK = "crack"
    UNDERCUT = "undercut"
    LACK_OF_FUSION = "lack_of_fusion"
    LACK_OF_PENETRATION = "lack_of_penetration"


unicode_to_label_mapping = {
    "\u6c14\u5b54": LabelNames.POROSITY,
    "\u5939\u6e23": LabelNames.INCLUSION,
    "\u5939\u94a8": LabelNames.INCLUSION,
    "\u88c2\u7eb9": LabelNames.CRACK,
    "\u54ac\u8fb9": LabelNames.UNDERCUT,
    "\u672a\u7194\u5408": LabelNames.LACK_OF_FUSION,
    "\u672a\u710a\u900f": LabelNames.LACK_OF_PENETRATION,
}


symbol_to_label_mapping = {
    "气孔": LabelNames.POROSITY,
    "夹渣": LabelNames.INCLUSION,
    "夹钨": LabelNames.INCLUSION,
    "裂纹": LabelNames.CRACK,
    "咬边": LabelNames.UNDERCUT,
    "未熔合": LabelNames.LACK_OF_FUSION,
    "未焊透": LabelNames.LACK_OF_PENETRATION,
}

label_to_color_mapping = {
    LabelNames.POROSITY: 1,
    LabelNames.INCLUSION: 2,
    LabelNames.CRACK: 3,
    LabelNames.UNDERCUT: 4,
    LabelNames.LACK_OF_FUSION: 5,
    LabelNames.LACK_OF_PENETRATION: 6,
}
