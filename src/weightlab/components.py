"""Analytic, bounded MLP column updates; no target training."""
import numpy as np

def protected_direction(d,h,beta=10.):
    moment=float(np.mean(h*h))
    if moment<=0:return d.copy()
    c=h.T@h/(len(h)*moment)
    return np.linalg.solve(np.eye(len(d))+beta*c,d)

def coefficients(weights,arrays,count=64):
    scores=[];directions={};norm2=0.
    for layer,w in weights.items():
        v=arrays[f'v{layer}'].astype(np.float64);u=v/np.linalg.norm(v);directions[layer]=u
        mu=np.concatenate([arrays[f'h{b}{layer}'] for b in ['pos','neg','plain','protect']]).mean(0)
        dots=w.T@u;norms=np.sqrt(np.sum(w*w,axis=0,dtype=np.float64));g=np.divide(dots*np.sign(mu),norms,out=np.zeros_like(dots),where=norms>0)
        scores.extend((float(abs(x)),layer,j,float(x),float(np.sign(dots[j]))) for j,x in enumerate(g))
        norm2+=float(np.sum(w*w,dtype=np.float64))
    scores.sort(key=lambda r:(-r[0],r[1],r[2]));chosen=scores[:count];tau=(scores[count-1][0]+scores[count][0])/2
    if tau<=0:raise ValueError('Zero sparse threshold')
    methods={k:{} for k in ['component','rank1','protected']}
    for layer in sorted(weights):
        sites=[r for r in chosen if r[1]==layer]
        if not sites:continue
        cols=np.array([r[2] for r in sites]);d=(arrays[f'hplain{layer}'].mean(0)-arrays[f'hprotect{layer}'].mean(0))[cols].astype(float)
        component=np.array([np.sign(r[3])*max(r[0]-tau,0)/tau*r[4] for r in sites])
        protection=protected_direction(d,arrays[f'hprotect{layer}'][:,cols].astype(float))
        for name,c in [('component',component),('rank1',d),('protected',protection)]:methods[name][layer]=(cols,c)
    return methods,directions,np.sqrt(norm2),{'tau':tau,'sites':[list(r) for r in chosen],'threshold_ties':scores[count-1][0]==scores[count][0]}
