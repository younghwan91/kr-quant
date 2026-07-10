/* 멀티투자자 OOS 파라미터서치. argv[1]=데이터dir, argv[2]=split_year(test 시작연도).
 * train(year<split)에서 최적 → test(year>=split) 검증. 과최적 방지.
 * 반절익절+본절스탑+러너 목표. 장중 스탑/목표(갭반영). 손익비 핵심. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define COST 0.0035
static int N, MAXH, SPLIT;
static int *yr; static double *ea, *path;
static void load(const char*dir){
    char p[512]; FILE*f;
    snprintf(p,512,"%s/meta.txt",dir); f=fopen(p,"r"); if(!f){fprintf(stderr,"no %s\n",p);exit(1);} if(fscanf(f,"%d %d",&N,&MAXH)!=2){exit(1);} fclose(f);
    yr=malloc(sizeof(int)*N); ea=malloc(sizeof(double)*2*N); path=malloc(sizeof(double)*(size_t)N*MAXH*4);
    snprintf(p,512,"%s/year.i32",dir); f=fopen(p,"rb"); if(fread(yr,sizeof(int),N,f)!=(size_t)N){} fclose(f);
    snprintf(p,512,"%s/ea.f64",dir); f=fopen(p,"rb"); if(fread(ea,sizeof(double),2*N,f)!=(size_t)2*N){} fclose(f);
    snprintf(p,512,"%s/path.f64",dir); f=fopen(p,"rb"); if(fread(path,sizeof(double),(size_t)N*MAXH*4,f)!=(size_t)N*MAXH*4){} fclose(f);
}
static double sim_one(int i,double katr,double halfT,double runT,int*valid){
    double E=ea[2*i],A=ea[2*i+1];
    if(!isfinite(E)||!isfinite(A)||A<=0||E<=0){*valid=0;return 0;}
    double R=katr*A,stop=E-R,th=E+halfT*R,tr=E+runT*R;
    int half=0,ex=0; double realized=0,ret=0,clast=NAN; size_t b0=(size_t)i*MAXH*4;
    for(int j=0;j<MAXH;j++){ size_t b=b0+(size_t)j*4;
        double o=path[b],h=path[b+1],l=path[b+2],c=path[b+3];
        if(!isfinite(c))continue; clast=c;
        if(isfinite(o)&&o<=stop){ret=realized+(half?0.5:1.0)*(o/E-1);ex=1;break;}
        if(isfinite(l)&&l<=stop){ret=realized+(half?0.5:1.0)*(stop/E-1);ex=1;break;}
        if(halfT>0&&!half&&isfinite(h)&&h>=th){realized=0.5*(th/E-1);half=1;stop=E;}
        if(isfinite(h)&&h>=tr){ret=realized+(half?0.5:1.0)*(tr/E-1);ex=1;break;}
    }
    if(!ex){ if(!isfinite(clast)){*valid=0;return 0;} ret=realized+(half?0.5:1.0)*(clast/E-1); }
    *valid=1; return ret-2*COST;
}
typedef struct{ double mean,payoff,t,win; int n; } Stat;
typedef struct{ Stat tr,te; double katr,halfT,runT; } Res;
static Res eval(double katr,double halfT,double runT){
    double s[2]={0,0},sq[2]={0,0},sw[2]={0,0},sl[2]={0,0}; int n[2]={0,0},nw[2]={0,0},nl[2]={0,0};
    for(int i=0;i<N;i++){ int v; double r=sim_one(i,katr,halfT,runT,&v); if(!v)continue;
        int g=(yr[i]>=SPLIT)?1:0; s[g]+=r; sq[g]+=r*r; n[g]++;
        if(r>0){sw[g]+=r;nw[g]++;} else if(r<0){sl[g]+=r;nl[g]++;} }
    Res R; R.katr=katr;R.halfT=halfT;R.runT=runT;
    for(int g=0;g<2;g++){ Stat*S=g?&R.te:&R.tr; S->n=n[g];
        S->mean=n[g]?s[g]/n[g]:0; double var=n[g]?sq[g]/n[g]-S->mean*S->mean:0,sd=var>0?sqrt(var):0;
        S->t=(n[g]&&sd>0)?S->mean/sd*sqrt((double)n[g]):0; S->win=n[g]?100.0*nw[g]/n[g]:0;
        S->payoff=(nw[g]&&nl[g])?(sw[g]/nw[g])/(-(sl[g]/nl[g])):NAN; }
    return R;
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: sim2 dir split_year\n");return 1;}
    load(argv[1]); SPLIT=atoi(argv[2]);
    double katrs[]={1.0,1.5,2.0,2.5,3.0},halfs[]={0,1,2,3},runs[]={5,8,12,20};
    Res best; int have=0,breadth=0,total=0;
    for(int a=0;a<5;a++)for(int b=0;b<4;b++)for(int c=0;c<4;c++){
        double kt=katrs[a],hT=halfs[b],rT=runs[c]; if(hT>0&&rT<=hT)continue;
        Res R=eval(kt,hT,rT); total++;
        if(R.tr.mean>0&&R.te.mean>0)breadth++;
        /* train 신뢰성(t) 최고 & train 기대값 양수 선택 → test로 검증 */
        if(R.tr.mean>0 && (!have || R.tr.t>best.tr.t)){ best=R; have=1; }
    }
    const char*nm=strrchr(argv[1],'/'); nm=nm?nm+1:argv[1];
    if(!have){ printf("%-12s (train 양수 조합 없음)\n",nm); return 0; }
    printf("%-12s | train n=%d | 선택 %.1fATR/반절%.0fR/러너%.0fR | TRAIN 평균%+.2f%% 손익비%.2f t%+.1f | TEST 평균%+.2f%% 손익비%.2f t%+.1f 승률%.0f%% | 견고%d/%d\n",
        nm, best.tr.n, best.katr,best.halfT,best.runT,
        best.tr.mean*100,best.tr.payoff,best.tr.t,
        best.te.mean*100,best.te.payoff,best.te.t,best.te.win, breadth,total);
    return 0;
}
