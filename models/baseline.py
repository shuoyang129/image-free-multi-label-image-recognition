import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from tqdm import tqdm
from PIL import Image
import numpy as np


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self,
                prompts,
                tokenized_prompts,
                if_embedding=True,
                if_sequence=False):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(
                self.dtype)  # [batch_size, n_ctx, d_model]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]

        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]),
                  tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.Caption.N_CTX
        ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.graph_prompt = 0
        self.graph_prompt_double = 0
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if cfg.DATASET.NAME in ["coco", "voc2007"]:
            self.co_cls =  [["aeroplane"],["parking meter"],["toilet"],["car","truck"],
                            ["laptop","tvmonitor","cell phone","cat","bed","remote","sofa","book"],
                            ["skis","snowboard"],["traffic light","stop sign","fire hydrant","train","bus","motorbike","bicycle","umbrella"],
                            ["person","bench","tie","dog","suitcase","backpack","handbag"],
                            ["fork","wine glass","knife","spoon","bowl","diningtable","bottle","cup"],
                            ["mouse","keyboard"], ["broccoli","carrot"],
                            ["surfboard","frisbee","skateboard","kite","horse","elephant","boat","bird"],
                            ["donut","pizza","teddy bear","scissors","chair","vase","pottedplant","clock"],
                            ["hair drier","sink","toothbrush"],[ "microwave","oven","toaster","refrigerator"],
                            ["sheep","cow"],["orange","banana","apple"],["giraffe","bear","zebra"],
                            ["tennis racket","baseball glove","sports ball","baseball bat"],
                            ["hot dog","sandwich","cake"]]
            self.co_graph = [["aeroplane","parking meter","car","truck","traffic light","stop sign",
                            "fire hydrant","train","bus","motorbike","bicycle","umbrella",
                            "person","bench","tie","dog","suitcase","backpack","handbag"], 
                            ["toilet","hair drier","sink","toothbrush"], 
                            ["laptop","tvmonitor","cell phone","cat","bed","remote","sofa","book",
                            "mouse","keyboard"], ["skis","snowboard"], 
                            ["fork","wine glass","knife","spoon","bowl","diningtable","bottle","cup",
                            "broccoli","carrot","donut","pizza","teddy bear","scissors","chair","vase","pottedplant","clock",
                            "microwave","oven","toaster","refrigerator","orange","banana","apple",
                            "hot dog","sandwich","cake"],
                            ["surfboard","frisbee","skateboard","kite","horse","elephant","boat","bird"], 
                            ["sheep","cow"], ["giraffe","bear","zebra"], 
                            ["tennis racket","baseball glove","sports ball","baseball bat"]]
            file = open("./data/coco-2014/classes.txt","r")

        elif cfg.DATASET.NAME == "nus_wide_zsl":
            self.co_cls = [["book"],["computer"],["fish"],["military"],
                        ["swimmers"],["tattoo"],["airport","plane"],
                        ["coral","whales"],["fire","cityscape","bridge","nighttime","cars","street","road","vehicle"],
                        ["food","leaf"],["map","flags","police","protest"],
                        ["tower","town","buildings","window","castle","house","earthquake","sign"],
                        ["cat","zebra","elk","birds","fox","tiger","bear","dog"],
                        ["animal","statue","flowers","sports","toy","person","beach","ocean"],
                        ["soccer","cow","horses"],["surf","glacier","temple","boats","harbor","wedding","garden","rocks"],
                        ["rainbow","reflection","clouds","sky","lake","water","tree","grass","plants"],
                        ["railroad","train"],
                        ["waterfall","moon","mountain","sunset","sand","sun","frost","snow","valley"],
                        ["dancing","running"]
                        ]
            self.co_graph = [["book"],["computer"],["fish","coral","whales"],
                        ["military","airport","plane"],["swimmers","cat","zebra","elk","birds","fox","tiger","bear","dog","animal","statue","flowers","sports","toy","person","beach","ocean","soccer","cow","horses","dancing","running"],
                        ["fire","cityscape","bridge","nighttime","cars","street","road","vehicle","food","leaf","tower","town","buildings","window","castle","house","earthquake","sign","surf","glacier","temple","boats","harbor","wedding","garden","rocks",
                        "rainbow","reflection","clouds","sky","lake","water","tree","grass","plants","railroad","train","waterfall","moon","mountain","sunset","sand","sun","frost","snow","valley"],
                        ["tattoo"],["map","flags","police","protest"]
                        ]
            file = open("./data/nus_wide/annotations/Tag_all/all_labels.txt","r")

        # elif cfg.DATASET.NAME == "voc2007":
        #     self.co_cls =  [["aeroplane"],["car"],
        #                     ["tvmonitor","cat","sofa"],["train","bus","motorbike","bicycle"],
        #                     ["person","dog"],["diningtable","bottle"],["horse","boat","bird"],
        #                     ["chair","pottedplant"],["sheep","cow"]]
        #     self.co_graph = [["aeroplane","car","train","bus","motorbike","bicycle","person","dog"],  
        #                     ["tvmonitor","cat","sofa"], 
        #                     ["diningtable","bottle","chair","pottedplant"],
        #                     ["horse","boat","bird"], 
        #                     ["sheep","cow"]]
        #     file = open("./data/VOCdevkit/VOC2007/classes.txt","r")

        cls_orig = []
        dict_cls = {}
        dict_graph = {}
        for line in file.readlines():
            cls_orig.append(line.strip())
        for i in cls_orig:
            for j in range(len(self.co_cls)):
                if i in self.co_cls[j]:
                    dict_cls[i] = j
        for i in cls_orig:
            for j in range(len(self.co_graph)):
                if i in self.co_graph[j]:
                    dict_graph[i] = j
        self.dict_cls = dict_cls
        self.dict_graph = dict_graph
        self.cls_orig = cls_orig
        
        n_co_cls = len(self.co_cls)
        n_co_graph = len(self.co_graph)
        
        n_ctx_all = 16
        n_ctx_co_cls = 8
        n_ctx_co_graph = 4
        n_ctx_single = 4
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            ctx_vectors_double = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx = nn.Parameter(ctx_vectors)
            elif cfg.TRAINER.Caption.GRAPH_PROMPT:
                self.graph_prompt = 1
                print("Initializing graph prompt")              
                
                ctx_vectors_single = torch.empty(n_cls, n_ctx_single, ctx_dim, dtype=dtype)
                ctx_vectors_all = torch.empty(1, n_ctx_all, ctx_dim, dtype=dtype)
                ctx_vectors_cls = torch.empty(n_co_cls, n_ctx_co_cls, ctx_dim, dtype=dtype)
                ctx_vectors_graph = torch.empty(n_co_graph, n_ctx_co_graph, ctx_dim, dtype=dtype)
                
                nn.init.normal_(ctx_vectors_single, std=0.02)
                nn.init.normal_(ctx_vectors_all, std=0.02)
                nn.init.normal_(ctx_vectors_cls, std=0.02)
                nn.init.normal_(ctx_vectors_graph, std=0.02)
                self.ctx_single = nn.Parameter(ctx_vectors_single)
                self.ctx_all = nn.Parameter(ctx_vectors_all)
                self.ctx_cls = nn.Parameter(ctx_vectors_cls)
                self.ctx_graph = nn.Parameter(ctx_vectors_graph)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx = nn.Parameter(ctx_vectors)

            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific double contexts")
                ctx_vectors_double = torch.empty(n_cls,n_ctx,ctx_dim,dtype=dtype)
                nn.init.normal_(ctx_vectors_double, std=0.02)
                self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
            elif cfg.TRAINER.Caption.GRAPH_PROMPT_DOUBLE:
                self.graph_prompt_double = 1
                ctx_vectors_double_single = torch.empty(n_cls, n_ctx_single, ctx_dim, dtype=dtype)
                ctx_vectors_double_all = torch.empty(1, n_ctx_all, ctx_dim, dtype=dtype)
                ctx_vectors_double_cls = torch.empty(n_co_cls, n_ctx_co_cls, ctx_dim, dtype=dtype)
                ctx_vectors_double_graph = torch.empty(n_co_graph, n_ctx_co_graph, ctx_dim, dtype=dtype)
                
                nn.init.normal_(ctx_vectors_double_single, std=0.02)
                nn.init.normal_(ctx_vectors_double_all, std=0.02)
                nn.init.normal_(ctx_vectors_double_cls, std=0.02)
                nn.init.normal_(ctx_vectors_double_graph, std=0.02)
                self.ctx_double_single = nn.Parameter(ctx_vectors_double_single)
                self.ctx_double_all = nn.Parameter(ctx_vectors_double_all)
                self.ctx_double_cls = nn.Parameter(ctx_vectors_double_cls)
                self.ctx_double_graph = nn.Parameter(ctx_vectors_double_graph)
            else:
                print("Initializing a generic context")
                ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors_double, std=0.02)
                self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat(
            [clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(
                tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls",embedding_nocls[:, 1 + n_ctx:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.Caption.CLASS_TOKEN_POSITION

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        if self.graph_prompt:
            ctx_single = self.ctx_single
            ctx_all = self.ctx_all.repeat(self.n_cls,1,1)
            ctx_cls = self.ctx_cls
            ctx_cls_f = torch.stack([ctx_cls[self.dict_cls[i]] for i in self.cls_orig],dim=0)
            ctx_graph = self.ctx_graph
            ctx_graph_f = torch.stack([ctx_graph[self.dict_graph[i]] for i in self.cls_orig],dim=0)
            ctx = torch.cat([ctx_all,ctx_cls_f,ctx_graph_f,ctx_single],dim=1)
        else:    
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.graph_prompt_double:
            ctx_double_single = self.ctx_double_single
            ctx_double_all = self.ctx_double_all.repeat(self.n_cls,1,1)
            ctx_double_cls = self.ctx_double_cls
            ctx_double_cls_f = torch.stack([ctx_double_cls[self.dict_cls[i]] for i in self.cls_orig],dim=0)
            ctx_double_graph = self.ctx_double_graph
            ctx_double_graph_f = torch.stack([ctx_double_graph[self.dict_graph[i]] for i in self.cls_orig],dim=0)
            ctx_double = torch.cat([ctx_double_all,ctx_double_cls_f,ctx_double_graph_f,ctx_double_single],dim=1)
        else:
            ctx_double = self.ctx_double
            if ctx_double.dim() == 2:
                ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,  # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # print(prompts.shape, prompts_neg.shape, "baseline.py line 248")
        return prompts, prompts_neg, self.temperature, self.spatial_T, self.ranking_scale


class Baseline(nn.Module):
    def __init__(self,
                 cfg,
                 classnames,
                 clip_model,
                 return_interm_layers=False):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.model = clip_model

        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3"
            }
        else:
            return_layers = {"layer4": "0"}
        
        self.visual_encoder = IntermediateLayerGetter(self.model.visual,
                                                      return_layers)
        self.positional_embedding = self.model.visual.attnpool.positional_embedding[
            1::]
        self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.model.visual.attnpool.c_proj.bias

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        
        text_features_hand = []
        if cfg.DATASET.NAME in ["coco", "voc2007"]:
            filename = './data/coco-2014/hand_prompt.txt'
        elif cfg.DATASET.NAME == "nus_wide_zsl":
            filename = './data/nus_wide/hand_prompt_n.txt'
        # elif cfg.DATASET.NAME == "voc2007":
        #     filename = './data/VOCdevkit/VOC2007/hand_prompt.txt'
        with open(filename, 'r') as f:
            lines = f.readlines() 
        for line in tqdm(lines):              
            captions = line.split("#####")
            caption = captions[0]
            try:
                caption = torch.cat([clip.tokenize(caption)])
            except:
                continue
            prompt_feat = self.text_encoder(caption,
                                            None,
                                            if_embedding=False,
                                            if_sequence=True)
            prompt_feature_ = prompt_feat[torch.arange(prompt_feat.shape[0]),
                                        caption.argmax(dim=-1)]  # BD
            text_features_hand.append(prompt_feature_)
        self.text_features_hand = torch.cat(text_features_hand,dim=0).cuda()

    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x
        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)

        return x

    def forward(self, image=None, captions=None, if_test=False):
        if if_test:
            image_feat = self.encode_image(image)
            b, c, h, w = image_feat.shape
            x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)

            x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
            x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
            image_features = x
            image_feature_ = self.model.visual.attnpool(image_feat)
            clip_feat = image_feature_ * 1

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner(
            )
            tokenized_prompts = self.tokenized_prompts

            text_features = self.text_encoder(prompts,
                                                  tokenized_prompts) + self.text_features_hand.detach()
            text_features_neg = self.text_encoder(prompts_double,
                                                  tokenized_prompts) + self.text_features_hand.detach()

            image_feature_ = image_feature_ / image_feature_.norm(dim=-1,
                                                                  keepdim=True)
            image_features = image_features / image_features.norm(dim=-1,
                                                                  keepdim=True)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(
                dim=-1, keepdim=True)
            
            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0
            logits_ = (logit_scale * image_feature_ @ text_features.t(
            )).softmax(dim=-1)  # B * C,  cls * C, = B * cls
            
            logits_neg = image_features @ text_features_neg.t(
            )  #  HW * B * C,  cls * C,  HW * B * cls
            
            tmp_scale = spatial_T.exp(
            ) if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image  # 5 #
            prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale,
                                                       dim=0)
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial,
                                     dim=0)
            logits_local = logits_local.softmax(dim=-1)
            return logits_, logits_local, logits_neg, image_features @ text_features.t(
            ), clip_feat
        else:
            cap_feat = self.text_encoder(captions,
                                        None,
                                        if_embedding=False,
                                        if_sequence=True)

            cap_feature_ = cap_feat[torch.arange(cap_feat.shape[0]),
                                    captions.argmax(dim=-1)]  # BD
            cap_features = cap_feat.permute(1, 0, 2)  # LBD

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner(
            )
            tokenized_prompts = self.tokenized_prompts

            text_features = self.text_encoder(prompts, tokenized_prompts) + self.text_features_hand.detach()
            text_features_neg = self.text_encoder(prompts_double,
                                                  tokenized_prompts) + self.text_features_hand.detach()

            cap_feature_ = cap_feature_ / cap_feature_.norm(dim=-1,
                                                                  keepdim=True)
            cap_features = cap_features / cap_features.norm(dim=-1,
                                                                  keepdim=True)
            
            noise_ = torch.randn_like(cap_feature_) * 0.1
            cap_feature_ = cap_feature_ + noise_
            noises = torch.randn_like(cap_features) * 0.1
            cap_features = cap_features + noises

            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(
                dim=-1, keepdim=True)

            # mask irrelavent tokens
            text_mask = (captions == 0).long() * (-10000)  # BL

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0
            logits_ = logit_scale * cap_feature_ @ text_features.t(
            )  # B * d,  cls * d, = B * cls
            logits_neg = cap_features @ text_features_neg.t(
            )  #  L * B * d,  cls * d,  L * B * cls
            logits_neg = logits_neg.permute(2, 1, 0) + text_mask[None, :, :]
            logits_neg = logits_neg.permute(2, 1, 0)  # L,B,cls

            tmp_scale = spatial_T.exp(
            ) if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text
            prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale,
                                                       dim=0)
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial,
                                     dim=0)

            return logits_, logits_local, text_features_neg, text_features

    @property
    def network_name(self):
        name = ''
        name += 'Baseline-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "visual" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'visual' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params


def baseline(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building baseline")
    model = Baseline(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(
            f"Multiple GPUs detected (n_gpus={device_count}), use all of them!"
        )
        model = nn.DataParallel(model)
    return model
