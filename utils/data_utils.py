'''
Utilites for data processing
'''
# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.
def fix_general_label_error(labels, slots):
    label_dict = labels
    GENERAL_TYPO = {
        # type
        "guest house":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports",
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall",
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east",
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre",
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
        # day
        "next friday":"friday", "monda": "monday",
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",
        }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no": label_dict[slot] = "north"
                elif label_dict[slot] == "we": label_dict[slot] = "west"
                elif label_dict[slot] == "cent": label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we": label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no": label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return slots

def normalize_ontology(ontology):
    keys = [k for k in ontology]
    for k in keys:
        for i in range(len(ontology[k])):
            ontology[k][i] = ontology[k][i].replace("do n't care", "dontcare")
            ontology[k][i] = ontology[k][i].replace("'s", " s")

        ontology[k.replace(" ","").lower() if ("book" not in k) else k.lower()] = ontology.pop(k)

    return ontology

'''
def collate_fn(data):
    batch_data = {}
    for field in data[0].__dataclass_fields__:
        batch_data[str(field)] = []
    for idx in range(len(data)):
        for field in data[idx].__dataclass_fields__:
            value = getattr(data, field)
            batch_data[str(field)]=value

    return batch_data
'''
