/* 앵커드 워크포워드 OOS. argv[1]=dir, argv[2]=first_test_year.
 * 각 test연도 TY: train=year<TY 에서 best(train t, train_mean>0) 선택 → test=year==TY 평가.
 * 재선택을 매년 반복(현실적). 연도별 OOS + 양수연도 집계. 반절/본절/러너 그리드. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define COST 0.0035
static int N, MAXH; static int *yr; static double *ea, *path;
static void load(const char*dir){ char p[512]; FILE*f;
    snprintf(p,512,"%s/meta.txt",dir); f=fopen(p,"r"); if(!f){fprintf(stderr,"no %s\n",p);exit(1);} if(fscanf(f,"%d %d",&N,&MAXH)!=2)exit(1); fclose(f);
    yr=malloc(sizeof(int)*N); ea=malloc(sizeof(double)*2*N); path=malloc(sizeof(double)*(size_t)N*MAXH*4);
    snprintf(p,512,"%s/year.i32",dir);f=fopen(p,"rb");if(fread(yr,sizeof(int),N,f)!=(size_t)N){}fclose(f);
    snprintf(p,512,"%s/ea.f64",dir);f=fopen(p,"rb");if(fread(ea,sizeof(double),2*N,f)!=(size_t)2*N){}fclose(f);
    snprintf(p,512,"%s/path.f64",dir);f=fopen(p,"rb");if(fread(path,sizeof(double),(size_t)N*MAXH*4,f)!=(size_t)N*MAXH*4){}fclose(f); }
static double sim_one(int i,double kt,double hT,double rT,int*v){
    double E=ea[2*i],A=ea[2*i+1]; if(!isfinite(E)||!isfinite(A)||A<=0||E<=0){*v=0;return 0;}
    double R=kt*A,stop=E-R,th=E+hT*R,tr=E+rT*R; int half=0,ex=0; double rz=0,ret=0,cl=NAN; size_t b0=(size_t)i*MAXH*4;
    for(int j=0;j<MAXH;j++){ size_t b=b0+(size_t)j*4; double o=path[b],h=path[b+1],l=path[b+2],c=path[b+3];
        if(!isfinite(c))continue; cl=c;
        if(isfinite(o)&&o<=stop){ret=rz+(half?0.5:1.0)*(o/E-1);ex=1;break;}
        if(isfinite(l)&&l<=stop){ret=rz+(half?0.5:1.0)*(stop/E-1);ex=1;break;}
        if(hT>0&&!half&&isfinite(h)&&h>=th){rz=0.5*(th/E-1);half=1;stop=E;}
        if(isfinite(h)&&h>=tr){ret=rz+(half?0.5:1.0)*(tr/E-1);ex=1;break;} }
    if(!ex){ if(!isfinite(cl)){*v=0;return 0;} ret=rz+(half?0.5:1.0)*(cl/E-1);} *v=1; return ret-2*COST; }
/* 특정 (train<TY) or (test==TY) 마스크로 통계 */
static void stats(double kt,double hT,double rT,int TY,int istest,
                  double*mean,double*payoff,double*t,int*n,double*win){
    double s=0,sq=0,sw=0,sl=0; int nn=0,nw=0,nl=0;
    for(int i=0;i<N;i++){ int inset = istest ? (yr[i]==TY) : (yr[i]<TY); if(!inset)continue;
        int v; double r=sim_one(i,kt,hT,rT,&v); if(!v)continue; s+=r;sq+=r*r;nn++;
        if(r>0){sw+=r;nw++;}else if(r<0){sl+=r;nl++;} }
    *n=nn; *mean=nn?s/nn:0; double var=nn?sq/nn-(*mean)*(*mean):0,sd=var>0?sqrt(var):0;
    *t=(nn&&sd>0)?(*mean)/sd*sqrt((double)nn):0; *win=nn?100.0*nw/nn:0;
    *payoff=(nw&&nl)?(sw/nw)/(-(sl/nl)):NAN; }
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: sim3_wf dir first_test_year\n");return 1;}
    load(argv[1]); int y0=atoi(argv[2]);
    const char*nm=strrchr(argv[1],'/'); nm=nm?nm+1:argv[1];
    double katrs[]={1.5,2.0,2.5,3.0},halfs[]={0,2,3},runs[]={8,12,20};
    printf("== %s 워크포워드 (train<TY → test=TY, 매년 재선택) ==\n",nm);
    int pos=0,tot=0; double tsum=0;
    for(int TY=y0;TY<=2026;TY++){
        double bkt=0,bhT=0,brT=0,btt=-1e9; int found=0;
        for(int a=0;a<4;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++){
            double kt=katrs[a],hT=halfs[b],rT=runs[c]; if(hT>0&&rT<=hT)continue;
            double m,po,tt,wi; int n; stats(kt,hT,rT,TY,0,&m,&po,&tt,&n,&wi);
            if(n<100)continue; if(m>0&&tt>btt){btt=tt;bkt=kt;bhT=hT;brT=rT;found=1;} }
        if(!found){ printf("  %d: (train 부족)\n",TY); continue; }
        double m,po,tt,wi; int n; stats(bkt,bhT,brT,TY,1,&m,&po,&tt,&n,&wi);
        if(n<20){ printf("  %d: (test 부족)\n",TY); continue; }
        printf("  %d: 선택 %.1fA/반절%.0fR/러너%.0fR | test n=%d 평균%+.2f%% 손익비%.2f t%+.1f 승률%.0f%%\n",
               TY,bkt,bhT,brT,n,m*100,po,tt,wi);
        tot++; if(m>0)pos++; tsum+=m*100;
    }
    printf("  >> 양수 test연도 %d/%d, 평균 test수익 %+.2f%%/트레이드\n",pos,tot,tot?tsum/tot:0);
    return 0;
}
