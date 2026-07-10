/* US-M04 피라미딩: 정찰대→신고가 재상승 성공증명 시 add-and-reduce(추가매수 시 스탑 상향, 총달러리스크 고정).
 * 물타기 금지(가격 오를 때만 추가). 6채널 경로(O,H,L,C,ma50,vol). 단일진입 vs 피라미딩 비교. walk-forward. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define COST 0.0035
#define MAXU 6
static int N,MAXH,NCH; static int *yr; static double *ea,*path;
static void load(const char*d){char p[512];FILE*f;
 snprintf(p,512,"%s/meta.txt",d);f=fopen(p,"r");if(!f)exit(1);if(fscanf(f,"%d %d %d",&N,&MAXH,&NCH)!=3)exit(1);fclose(f);
 yr=malloc(4*N);ea=malloc(16*N);path=malloc(8*(size_t)N*MAXH*NCH);
 snprintf(p,512,"%s/year.i32",d);f=fopen(p,"rb");if(fread(yr,4,N,f)!=(size_t)N){}fclose(f);
 snprintf(p,512,"%s/ea.f64",d);f=fopen(p,"rb");if(fread(ea,8,2*N,f)!=(size_t)2*N){}fclose(f);
 snprintf(p,512,"%s/path.f64",d);f=fopen(p,"rb");if(fread(path,8,(size_t)N*MAXH*NCH,f)!=(size_t)N*MAXH*NCH){}fclose(f);}
/* hard=손절%, step=추가매수 상승트리거%, K=추가간 최소일수, maxadd=최대추가수(0=단일), ma=50일트레일.
 * 반환: 총 R-멀티플(총손익/초기1유닛달러리스크). base_risk=hard*E. */
static double sim(int i,double hard,double step,int K,int maxadd,int ma,int*v){
 double E=ea[2*i];if(!isfinite(E)||E<=0){*v=0;return 0;}
 double ent[MAXU]; int U=1; ent[0]=E;
 double baserisk=hard*E;                       /* 1유닛 초기 달러리스크 */
 double stop=E-baserisk;                        /* = E*(1-hard) */
 double lastadd=E; int lastday=0, ex=0; double cl=NAN, exitpx=NAN;
 size_t b0=(size_t)i*MAXH*NCH;
 for(int j=0;j<MAXH;j++){size_t b=b0+(size_t)j*NCH;
  double o=path[b],h=path[b+1],l=path[b+2],c=path[b+3],m=path[b+4];
  if(!isfinite(c))continue;
  if(isfinite(o)&&o<=stop){exitpx=o;ex=1;break;}
  if(isfinite(l)&&l<=stop){exitpx=stop;ex=1;break;}
  /* 추가매수: 신고가 재상승(마지막추가가 대비 +step) + 자연조정(K일 경과) + 물타기금지(항상 상승방향) */
  if(U-1<maxadd && isfinite(h) && h>=lastadd*(1+step) && (j-lastday)>=K){
   double A=lastadd*(1+step); ent[U++]=A; lastadd=A; lastday=j;
   /* add-and-reduce: 총 달러리스크 = baserisk 유지 → stop = (Σentry - baserisk)/U */
   double se=0; for(int u=0;u<U;u++)se+=ent[u];
   double ns=(se-baserisk)/U; if(ns>stop)stop=ns;
  }
  if(ma&&isfinite(m)&&c<m&&c>stop){exitpx=c;ex=1;break;}
  cl=c;
 }
 if(!ex){ for(int j=MAXH-1;j>=0;j--){double c=path[b0+(size_t)j*NCH+3];if(isfinite(c)){exitpx=c;break;}} if(!isfinite(exitpx)){*v=0;return 0;} }
 /* 총손익(유닛당 비용차감) / baserisk = R멀티플 */
 double pnl=0,inv=0; for(int u=0;u<U;u++){ pnl+=(exitpx-ent[u]) - 2*COST*ent[u]; inv+=ent[u]; }
 *v=1; return pnl/baserisk;   /* R-멀티플 */
}
static void run(const char*lbl,double hd,double st,int K,int maxadd,int ma){
 double s=0,sq=0,sw=0,sl=0;int nn=0,nw=0,nl=0; double yrsum=0;int posy=0,ny=0;
 for(int TY=2018;TY<=2026;TY++){double ys=0;int yn=0;
  for(int i=0;i<N;i++){if(yr[i]!=TY)continue;int v;double r=sim(i,hd,st,K,maxadd,ma,&v);if(!v)continue;
   s+=r;sq+=r*r;nn++;ys+=r;yn++;if(r>0){sw+=r;nw++;}else if(r<0){sl+=r;nl++;}}
  if(yn>20){ys/=yn;yrsum+=ys;ny++;if(ys>0)posy++;}}
 double mn=nn?s/nn:0,po=(nw&&nl)?(sw/nw)/(-(sl/nl)):0;
 printf("%-30s 평균 %+.3fR 손익비%.2f 승률%.0f%% | 양수 %d/%d 연평균%+.3fR\n",
  lbl,mn,po,100.0*nw/nn,posy,ny,ny?yrsum/ny:0);
}
int main(int argc,char**argv){load(argv[1]);
 printf("== US-M04 피라미딩 (R-멀티플, 총달러리스크 고정) 하드손절8%% ==\n");
 run("단일진입(피라미딩無)",        0.08,0.05,5,0,1);
 run("피라미딩 step5%%/K5/최대2",    0.08,0.05,5,2,1);
 run("피라미딩 step5%%/K5/최대3",    0.08,0.05,5,3,1);
 run("피라미딩 step10%%/K10/최대2",  0.08,0.10,10,2,1);
 run("피라미딩 step10%%/K10/최대3",  0.08,0.10,10,3,1);
 run("피라미딩 step8%%/K7/최대4",    0.08,0.08,7,4,1);
 return 0;}
