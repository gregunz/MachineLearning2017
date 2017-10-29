
# coding: utf-8

# In[1]:


import numpy as np

from helpers import *
from functions import inv_log, mult, abs_dif
from preprocessing import replace_invalid, standardize
from feature_eng import build_poly
from cross_validation import cv_with_list
from predictions import predict_with_ridge


# In[2]:


y, x_brute_train, _ = load_csv_data("../data/train.csv")


# In[3]:


_, x_brute_test, indices_test = load_csv_data("../data/test.csv")


# In[4]:


y.shape, x_brute_train.shape, x_brute_test.shape


# In[5]:


train_size = x_brute_train.shape[0]
test_size = x_brute_test.shape[0]

train_size, test_size


# In[6]:


x_brute = np.concatenate((x_brute_train, x_brute_test))
x_brute.shape


# In[7]:


invalid_value = -999


# In[8]:


features_name = ["DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h","DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality","PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met","PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]


# In[9]:


PHI_features = [i for i, f in enumerate(features_name) if ("_phi" in f) and ("_phi_" not in f)]

PHI_features


# # Conditioning on features 22

# In[10]:


def verify_masks(masks):
    total = 0
    for mask in masks:
        num = mask.sum()
        print(num)
        total += num
    assert total == x_brute.shape[0]
    return len(masks)


# In[11]:


data_masks = [
    x_brute[:, 22] == 0,
    x_brute[:, 22] == 1,
    x_brute[:, 22] > 1
]
        
verify_masks(data_masks)


# ### Mask on Y

# In[12]:


ys = [y[mask[:train_size]] for mask in data_masks]

[y.shape for y in ys]


# ### Mask on X

# In[13]:


features_masks = [x_brute[m].std(axis=0) != 0 for m in data_masks]


# In[14]:


xs_brute = [x_brute[d_m][:, f_m] for d_m, f_m in zip(data_masks, features_masks)]

[x.shape for x in xs_brute]


# <br/><br/><br/>
# 
# # Data Preprocessing

# ## Replace by mean or most frequent

# In[15]:


xs_replace_invalid = [replace_invalid(x, ~(x == invalid_value), replace_by="mf") for x in xs_brute]

[x.shape for x in xs_replace_invalid]


# ## Remove Angles

# In[16]:


mask_phi_features = range_mask(30, PHI_features)


# In[17]:


xs_cleaned = [x[:, ~mask_phi_features[mask]] for x, mask in zip(xs_replace_invalid, features_masks)]

[x.shape for x in xs_cleaned]


# <br/><br/><br/>
# 
# # Features Engineering

# ## Features with log & Standardization

# In[18]:


xs_non_negative = [x - x.min(axis=0) for x in xs_cleaned]


# In[19]:


xs_standardized = [standardize(x) for x in xs_cleaned]

[f.shape for f in xs_standardized]


# In[20]:


xs_log = [standardize(np.log(1 + x)) for x in xs_non_negative]
    
[f.shape for f in xs_log]


# In[21]:


xs_inv_log = [standardize(inv_log(x)) for x in xs_non_negative]
    
[f.shape for f in xs_inv_log]


# ## Polynomial features

# ### Powers

# In[22]:


def create_poly_features(xs, degrees):
    return [build_poly(x, degree) for x, degree in zip(xs, degrees)]


# In[23]:


degrees_no_angles = [6, 10, 11]
poly_std = create_poly_features(xs_standardized, degrees_no_angles)

[x.shape for x in poly_std]


# In[24]:


degrees_log = [5, 5, 5]
poly_log = create_poly_features(xs_log, degrees_log)

[x.shape for x in poly_log]


# In[25]:


degrees_inv_log = [5, 5, 5]
poly_inv_log = create_poly_features(xs_inv_log, degrees_inv_log)

[x.shape for x in poly_inv_log]


# ### Roots

# In[26]:


def create_poly_roots_features(xs, degrees):
    return [build_poly(x, degree, roots=True)[:, x.shape[1]:] for x, degree in zip(xs, degrees)]


# In[27]:


degrees_roots_no_angles = [3, 3, 3]
poly_roots = create_poly_roots_features(xs_non_negative, degrees_roots_no_angles)
[x.shape for x in poly_roots]


# ### Powers + Roots

# In[28]:


features_poly = [np.concatenate(x, axis=1) for x in zip(poly_std, poly_roots, poly_log, poly_inv_log)]

[x.shape for x in features_poly]


# ## Combinations of features

# In[29]:


xs_mix = [np.concatenate(x, axis=1) for x in zip(xs_standardized, xs_log, xs_inv_log)]
    
[f.shape for f in xs_mix]


# In[30]:


def build_combinations(fn, xs):
    fn_combinations = [create_pairs(x.shape[1], x.shape[1]) for x in xs]
    print([len(c) for c in fn_combinations])
    return all_combinations_of(xs, fn, fn_combinations)
    


# In[31]:


features_mult = build_combinations(mult, xs_mix)

[x.shape for x in features_mult]


# In[32]:


features_abs_dif = build_combinations(abs_dif, xs_mix)

[x.shape for x in features_abs_dif]


# ## Constant features (ones)

# In[ ]:


features_ones = [np.ones(m.sum()).reshape((m.sum(), 1)) for m in data_masks]

[x.shape for x in features_ones]


# ## Concat all features

# In[ ]:


all_features = zip(
    features_ones,
    features_poly,
    features_mult,
    features_abs_dif,
)

features = [np.concatenate([f for f in list(fs) if len(f) > 0], axis=1) for fs in list(all_features)]

[f.shape for f in features]


# # Separating Training and Test data

# In[ ]:


xs_train, xs_test = separate_train(features, train_size, data_masks)

[(tr.shape, te.shape) for tr, te in zip(xs_train, xs_test)]


# # Cross validation

# In[ ]:


k_fold = 4
iters = 1

lambdas = [1e-04] * 3 #[1e-05, 1e-05, 1e-05]
seed = np.random.randint(10000)

scores = cv_with_list(ys, xs_train, lambdas, k_fold=k_fold, iters=iters, seed=seed, print_=True)

final_score = np.sum([score * x.shape[0] / train_size for score, x in zip(scores.mean(axis=0), xs_train)])
print("Final Test Error = {}".format(final_score * 100))


# # Submission

# In[ ]:


y_submission = predict_with_ridge(ys, xs_train, xs_test, lambdas, data_masks)


# In[ ]:


create_csv_submission(indices_test, y_submission, "submissions/pred27.csv")

