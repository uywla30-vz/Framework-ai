#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <algorithm>
namespace hwsb {
Tokenizer::Tokenizer() { initialize_base_vocab(); }
void Tokenizer::initialize_base_vocab() {
    id_to_token.clear(); token_to_id.clear(); merges.clear();
    for (int i=0; i<256; ++i) { std::vector<uint8_t> bt={(uint8_t)i}; id_to_token[i]=bt; token_to_id[bt]=i; }
    id_to_token[BOS_TOKEN]={}; id_to_token[EOS_TOKEN]={}; id_to_token[PAD_TOKEN]={};
}
void Tokenizer::train(const std::string& cp, int tvs) {
    std::ifstream f(cp, std::ios::binary); if(!f) return;
    std::vector<uint8_t> c((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int> t; for (uint8_t b:c) t.push_back(b);
    int cvs=256, nid=256;
    while(cvs<tvs){
        std::map<std::pair<int,int>,int> pc; for(size_t i=0;i+1<t.size();++i) pc[{t[i],t[i+1]}]++;
        if(pc.empty())break;
        auto bp=std::max_element(pc.begin(),pc.end(),[](const auto& a,const auto& b){return a.second<b.second;});
        if(bp->second<2)break;
        merges[bp->first]=nid; std::vector<uint8_t> nt=id_to_token[bp->first.first];
        nt.insert(nt.end(),id_to_token[bp->first.second].begin(),id_to_token[bp->first.second].end());
        id_to_token[nid]=nt; token_to_id[nt]=nid;
        std::vector<int> nts; for(size_t i=0;i<t.size();++i){
            if(i+1<t.size()&&t[i]==bp->first.first&&t[i+1]==bp->first.second){nts.push_back(nid);i++;}
            else nts.push_back(t[i]);
        }
        t=std::move(nts); nid++; cvs++;
    }
}
std::vector<int> Tokenizer::encode(const std::string& txt) const {
    std::vector<int> t; for(char c:txt) t.push_back((uint8_t)c);
    bool m=true; while(m){
        m=false; int mid=1000000; std::pair<int,int> bp;
        for(size_t i=0;i+1<t.size();++i){
            auto it=merges.find({t[i],t[i+1]}); if(it!=merges.end()&&it->second<mid){mid=it->second;bp=it->first;m=true;}
        }
        if(m){ std::vector<int> nts; for(size_t i=0;i<t.size();++i){
            if(i+1<t.size()&&t[i]==bp->first&&t[i+1]==bp->second){nts.push_back(mid);i++;}
            else nts.push_back(t[i]);
        } t=std::move(nts); }
    } return t;
}
std::string Tokenizer::decode(const std::vector<int>& ts) const {
    std::string s; for(int id:ts) if(id_to_token.count(id)) for(uint8_t b:id_to_token.at(id)) s+=(char)b;
    return s;
}
void Tokenizer::save(const std::string& vp, const std::string& mp) const {
    std::ofstream v(vp,std::ios::binary); for(const auto& [id,b]:id_to_token){
        v.write((char*)&id,4); int s=b.size(); v.write((char*)&s,4); v.write((char*)b.data(),s);
    }
    std::ofstream m(mp,std::ios::binary); for(const auto& [p,id]:merges){
        m.write((char*)&p.first,4); m.write((char*)&p.second,4); m.write((char*)&id,4);
    }
}
void Tokenizer::load(const std::string& vp, const std::string& mp) {
    initialize_base_vocab(); std::ifstream v(vp,std::ios::binary); if(v){
        while(v.peek()!=EOF){ int id,s; v.read((char*)&id,4); v.read((char*)&s,4); std::vector<uint8_t> b(s); v.read((char*)b.data(),s); id_to_token[id]=b; token_to_id[b]=id; }
    }
    std::ifstream m(mp,std::ios::binary); if(m){
        while(m.peek()!=EOF){ int p1,p2,id; m.read((char*)&p1,4); m.read((char*)&p2,4); m.read((char*)&id,4); merges[{p1,p2}]=id; }
    }
}
}
