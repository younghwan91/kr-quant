import numpy as np
D="btm/mnv_exit"
n,MAXH,NCH=[int(x) for x in open(f"{D}/meta.txt").readline().split()]
ea=np.fromfile(f"{D}/ea.f64").reshape(n,2); path=np.fromfile(f"{D}/path.f64").reshape(n,MAXH,NCH)
T=np.fromfile(f"{D}/t.i32",dtype=np.int32); nD=int(open(f"{D}/ndays.txt").read())
COST=0.0035; HARD=0.10
# 각 트레이드: 하드8%손절 + 50일선 트레일 exit → (exit_offset, ret_on_capital)
def exit_of(i):
    E=ea[i,0]; 
    if not np.isfinite(E) or E<=0: return None
    stop=E*(1-HARD)
    for j in range(MAXH):
        o,h,l,c,m=path[i,j,0],path[i,j,1],path[i,j,2],path[i,j,3],path[i,j,4]
        if not np.isfinite(c): continue
        if np.isfinite(o) and o<=stop: return j, o/E-1-2*COST
        if np.isfinite(l) and l<=stop: return j, stop/E-1-2*COST
    # 시간청산
    for j in range(MAXH-1,-1,-1):
        c=path[i,j,3]
        if np.isfinite(c): return j, c/E-1-2*COST
    return None
trades=[]  # (entry_day, exit_day, ret)
for i in range(n):
    e=exit_of(i)
    if e is None: continue
    off,ret=e; trades.append((int(T[i]), int(T[i]+off), ret))
trades.sort()
def run_portfolio2(maxpos, size):
    equity=1.0; eq_curve=np.full(nD,np.nan); eq_curve[0]=1.0; taken=[0]
    open_pos=[]  # (exit_day, ret, size_frac)
    ti=0; trades_sorted=trades
    # 진입 인덱스: 날짜별
    from collections import defaultdict
    byday=defaultdict(list)
    for (ed,xd,r) in trades_sorted: byday[ed].append((xd,r))
    for day in range(nD):
        # 청산
        still=[]
        for (xd,r,sz) in open_pos:
            if xd<=day: equity*= (1+sz*r)      # 청산 실현
            else: still.append((xd,r,sz))
        open_pos=still
        # 진입(first-come, 슬롯 여유시)
        for (xd,r) in byday.get(day,[]):
            if len(open_pos)<maxpos:
                open_pos.append((xd,r,size)); taken[0]+=1
        eq_curve[day]=equity
    eq=eq_curve.copy()
    # 통계
    cagr=eq[-1]**(252.0/nD)-1
    peak=np.maximum.accumulate(eq); dd=(eq/peak-1); mdd=dd.min()
    dr=np.diff(eq)/eq[:-1]; sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    return eq[-1],cagr,mdd,sharpe,taken[0]
rs=np.array([r for _,_,r in trades]); print(f"평균 트레이드수익 {rs.mean()*100:+.2f}%, 최악 {rs.min()*100:.0f}%, 승률 {(rs>0).mean()*100:.0f}%"); print(f"트레이드 {len(trades)}개, 기간 {nD}거래일(~{nD/252:.1f}년)")
print(f"{'동시보유':>6} {'종목비중':>7} {'최종자산':>7} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7}")
for maxpos,size in [(20,0.03),(40,0.02),(60,0.015),(100,0.01)]:
    fin,cagr,mdd,sh,nt=run_portfolio2(maxpos,size)
    print(f"{maxpos:>6} {size*100:>6.1f}% {fin:>7.2f}x {cagr*100:>+6.1f}% {mdd*100:>+6.0f}% {sh:>+6.2f}  taken={nt}/{len(trades)}")
