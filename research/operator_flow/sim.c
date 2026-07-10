/* 세력+돌파 트레이드 시뮬레이터 + 파라미터 그리드 서치 (C, 고속).
 * 반절 익절 + 본절(breakeven) 스탑 + 러너는 큰 목표까지. 장중 스탑/목표(갭반영).
 * 입력: bt/meta.txt(n MAXH), bt/year.i32, bt/ea.f64(E,A×n), bt/path.f64(n×MAXH×4 OHLC).
 * 손익비(payoff)가 핵심 지표. 기대값·ex-2025 양수 중 손익비 최고 추천. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define COST 0.0035

static int N, MAXH;
static int *yr;
static double *ea;    /* 2*N */
static double *path;  /* N*MAXH*4 */

static void load(void){
    FILE *f=fopen("bt/meta.txt","r"); fscanf(f,"%d %d",&N,&MAXH); fclose(f);
    yr=malloc(sizeof(int)*N); ea=malloc(sizeof(double)*2*N);
    path=malloc(sizeof(double)*(size_t)N*MAXH*4);
    f=fopen("bt/year.i32","rb"); fread(yr,sizeof(int),N,f); fclose(f);
    f=fopen("bt/ea.f64","rb"); fread(ea,sizeof(double),2*N,f); fclose(f);
    f=fopen("bt/path.f64","rb"); fread(path,sizeof(double),(size_t)N*MAXH*4,f); fclose(f);
}

/* 한 트레이드 시뮬 → net 수익률 반환. valid=0이면 스킵. */
static double sim_one(int i,double katr,double halfT,double runT,int *valid){
    double E=ea[2*i], A=ea[2*i+1];
    if(!isfinite(E)||!isfinite(A)||A<=0||E<=0){*valid=0;return 0;}
    double R=katr*A, stop=E-R, th=E+halfT*R, tr=E+runT*R;
    int half=0, exited=0; double realized=0.0, ret=0.0, clast=NAN;
    size_t b0=(size_t)i*MAXH*4;
    for(int j=0;j<MAXH;j++){
        size_t b=b0+(size_t)j*4;
        double o=path[b],h=path[b+1],l=path[b+2],c=path[b+3];
        if(!isfinite(c)) continue;
        clast=c;
        if(isfinite(o)&&o<=stop){ ret=realized+(half?0.5:1.0)*(o/E-1); exited=1; break; } /* 갭하락 시가 */
        if(isfinite(l)&&l<=stop){ ret=realized+(half?0.5:1.0)*(stop/E-1); exited=1; break; } /* 손절/본절 */
        if(halfT>0 && !half && isfinite(h) && h>=th){ realized=0.5*(th/E-1); half=1; stop=E; } /* 반절+본절 */
        if(isfinite(h) && h>=tr){ ret=realized+(half?0.5:1.0)*(tr/E-1); exited=1; break; }    /* 목표도달 */
    }
    if(!exited){
        if(!isfinite(clast)){*valid=0;return 0;}
        ret=realized+(half?0.5:1.0)*(clast/E-1); /* 시간청산 */
    }
    *valid=1; return ret-2*COST;
}

typedef struct { double mean,win,payoff,t,exmean,expayoff; int n,posyr; } Res;

static Res eval(double katr,double halfT,double runT){
    double sum=0,sq=0,sw=0,sl=0; int n=0,nw=0,nl=0;
    double esum=0,esw=0,esl=0; int en=0,enw=0,enl=0;
    double ysum[16]={0}; int yn[16]={0};
    for(int i=0;i<N;i++){
        int v; double r=sim_one(i,katr,halfT,runT,&v);
        if(!v) continue;
        sum+=r; sq+=r*r; n++;
        if(r>0){sw+=r;nw++;} else if(r<0){sl+=r;nl++;}
        int yi=yr[i]-2015; if(yi>=0&&yi<16){ysum[yi]+=r;yn[yi]++;}
        if(yr[i]!=2025){ esum+=r; en++; if(r>0){esw+=r;enw++;} else if(r<0){esl+=r;enl++;} }
    }
    Res R; R.n=n;
    R.mean= n? sum/n:0;
    double var= n? sq/n - R.mean*R.mean:0; double sd= var>0? sqrt(var):0;
    R.t= (n&&sd>0)? R.mean/sd*sqrt((double)n):0;
    R.win= n? 100.0*nw/n:0;
    R.payoff= (nw&&nl)? (sw/nw)/(-(sl/nl)):NAN;
    R.exmean= en? esum/en:0;
    R.expayoff= (enw&&enl)? (esw/enw)/(-(esl/enl)):NAN;
    int py=0; for(int k=0;k<16;k++) if(yn[k]>0&&ysum[k]>0) py++;
    R.posyr=py;
    return R;
}

int main(void){
    load();
    printf("세력+돌파 반절/본절 그리드 (n=%d MAXH=%d, 편도%.2f%%)\n",N,MAXH,COST*100);
    printf("%5s %6s %6s %6s %7s %5s %7s %6s | ex25 평균/손익비 양수년\n",
           "손절","반절","러너","평균%","승률","손익비","t","");
    double katrs[]={1.0,1.5,2.0,2.5,3.0}; int nk=5;
    double halfs[]={0,1,2,3}; int nh=4;
    double runs[]={5,8,12,20}; int nr=4;
    Res best; double bestpay=-1; char bestlbl[64]="";
    for(int a=0;a<nk;a++)for(int b=0;b<nh;b++)for(int c=0;c<nr;c++){
        double katr=katrs[a],hT=halfs[b],rT=runs[c];
        if(hT>0 && rT<=hT) continue;
        Res R=eval(katr,hT,rT);
        char hl[16]; if(hT==0) snprintf(hl,16,"없음"); else snprintf(hl,16,"%.0fR",hT);
        printf("%4.1fA %6s %5.0fR %+6.2f %4.0f%% %6.2f %+6.1f | %+.2f%%/%5.2f  %d/10\n",
               katr,hl,rT,R.mean*100,R.win,R.payoff,R.t,R.exmean*100,R.expayoff,R.posyr);
        if(R.mean>0 && R.exmean>0 && R.payoff>bestpay){
            bestpay=R.payoff; best=R;
            snprintf(bestlbl,64,"손절%.1fATR/반절%s/러너%.0fR",katr,hl,rT);
        }
    }
    printf("\n★추천(기대값·ex2025 양수 中 손익비최고): %s\n",bestlbl);
    printf("  평균%+.2f%%/트레이드 승률%.0f%% 손익비%.2f t%+.1f | ex-2025 %+.2f%%/손익비%.2f 양수%d/10\n",
           best.mean*100,best.win,best.payoff,best.t,best.exmean*100,best.expayoff,best.posyr);
    return 0;
}
