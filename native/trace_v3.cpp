// One resident model; line-delimited requests, bounded native tensor capture.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
struct Trace {
    bool enabled=false;
    std::regex filter;
    std::filesystem::path folder;
    json captures=json::object();
    json intervention;
};
static bool callback(ggml_tensor *t,bool ask,void *user){
    auto &s=*static_cast<Trace*>(user);
    bool edit=s.intervention.is_object() && s.intervention.value("tensor","")==t->name;
    bool wanted=(s.enabled && std::regex_match(t->name,s.filter)) || edit;
    if(ask)return wanted;
    if(!wanted)return true;
    if(t->type!=GGML_TYPE_F32 || t->ne[2]!=1 || t->ne[3]!=1 || t->nb[0]!=sizeof(float))
        throw std::runtime_error("Only finite F32 matrix traces are supported: "+std::string(t->name));
    const size_t n=t->ne[0],offset=(t->ne[1]-1)*t->nb[1];
    std::vector<float> values(n);
    ggml_backend_tensor_get(t,values.data(),offset,n*sizeof(float));
    if(edit){
        auto direction=s.intervention.at("vector").get<std::vector<float>>();
        if(direction.size()!=n)throw std::runtime_error("Intervention width mismatch");
        float scale=s.intervention.value("scale",1.0f);
        for(size_t i=0;i<n;i++)values[i]+=scale*direction[i];
        ggml_backend_tensor_set(t,values.data(),offset,n*sizeof(float));
    }
    for(float v:values)if(!std::isfinite(v))throw std::runtime_error("Nonfinite trace");
    std::string name=t->name;
    if(!std::regex_match(name,std::regex("[A-Za-z0-9_.-]+")))throw std::runtime_error("Unsafe tensor name");
    auto path=s.folder/(name+".f32");
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float));out.flush();
    if(!out)throw std::runtime_error("Cannot save trace");
    s.captures[name]={{"path",path.string()},{"elements",n},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},
                      {"last_column_offset_bytes",offset},{"op",ggml_op_name(t->op)}};
    return true;
}
int main(int argc,char **argv){
    try{
        if(argc!=2)throw std::runtime_error("usage: weightlab-native CONFIG.json");
        json cfg;std::ifstream(argv[1])>>cfg;
        ggml_backend_load_all_from_path(cfg.at("library_dir").get<std::string>().c_str());
        llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=cfg.value("gpu_layers",0);
        auto *model=llama_model_load_from_file(cfg.at("model").get<std::string>().c_str(),mp);
        if(!model)throw std::runtime_error("Model load failed");
        Trace trace;trace.enabled=cfg.value("capture",true);
        trace.filter=std::regex(cfg.value("filter","result_norm|ffn_out-[0-9]+|ffn_silu-[0-9]+"));
        auto cp=llama_context_default_params();
        cp.n_ctx=cfg.value("context",2048);cp.n_batch=cp.n_ctx;cp.n_ubatch=cp.n_ctx;
        cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;
        if(cfg.value("callback",true)){cp.cb_eval=callback;cp.cb_eval_user_data=&trace;}
        auto *ctx=llama_init_from_model(model,cp);
        if(!ctx)throw std::runtime_error("Context load failed");
        const auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
        std::cout<<json({{"event","ready"},{"vocab",nv}}).dump()<<std::endl;
        std::string line;
        while(std::getline(std::cin,line)){
            if(line.empty())continue;
            json req=json::parse(line),answer={{"id",req.at("id")}};
            try{
                trace.enabled=cfg.value("capture",true);trace.captures=json::object();trace.intervention=req.value("intervention",json());
                trace.folder=req.at("trace_dir").get<std::string>();
                std::filesystem::create_directories(trace.folder);
                std::string prompt=req.at("prompt");
                int n=llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
                if(n>=0)throw std::runtime_error("Unexpected tokenize length");
                std::vector<llama_token> tokens(-n);
                n=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),true,true);
                if(n<=0 || n>int(cp.n_ctx))throw std::runtime_error("Prompt length outside context");
                tokens.resize(n);llama_memory_clear(llama_get_memory(ctx),true);
                auto begin=std::chrono::steady_clock::now();
                auto batch=llama_batch_get_one(tokens.data(),n);
                if(llama_decode(ctx,batch)!=0)throw std::runtime_error("Decode failed");
                const float *logits=llama_get_logits_ith(ctx,-1);
                if(!logits)throw std::runtime_error("Missing logits");
                std::vector<int> indices(nv);for(int i=0;i<nv;i++)indices[i]=i;
                std::partial_sort(indices.begin(),indices.begin()+64,indices.end(),[&](int a,int b){return logits[a]>logits[b];});
                json top=json::array();for(int j=0;j<64;j++)top.push_back({{"id",indices[j]},{"logit",logits[indices[j]]}});
                auto initial_captures=trace.captures;
                auto logpath=trace.folder/"logits.f32";std::ofstream out(logpath,std::ios::binary);
                out.write(reinterpret_cast<const char*>(logits),nv*sizeof(float));out.close();
                std::vector<llama_token> generated;
                std::string completion;std::string finish="length";
                int maximum=req.value("max_tokens",0);
                if(maximum<0 || maximum>256)throw std::runtime_error("Generation bound");
                trace.enabled=false;trace.folder=trace.folder/"generation";
                if(maximum)std::filesystem::create_directories(trace.folder);
                for(int step=0;step<maximum;step++){
                    const float * current=llama_get_logits_ith(ctx,-1);
                    auto token=llama_token(std::max_element(current,current+nv)-current);
                    generated.push_back(token);
                    if(llama_vocab_is_eog(vocab,token)){finish="stop";break;}
                    std::vector<char> piece(256);
                    int length=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);
                    if(length<0){piece.resize(-length);length=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);}
                    if(length<0)throw std::runtime_error("Token piece");
                    completion.append(piece.data(),length);
                    if(step+1<maximum && llama_decode(ctx,llama_batch_get_one(&token,1))!=0)throw std::runtime_error("Generation decode");
                }
                answer.update({{"status","complete"},{"tokens",tokens},{"prompt_tokens",n},{"generated_tokens",generated.size()},{"completion",completion},{"generated_ids",generated},{"finish_reason",finish},
                    {"top64",top},{"logits_path",logpath.string()},{"captures",initial_captures},
                    {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}});
            }catch(const std::exception &e){answer.update({{"status","failed"},{"error",e.what()}});}
            std::cout<<answer.dump()<<std::endl;
        }
        llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
