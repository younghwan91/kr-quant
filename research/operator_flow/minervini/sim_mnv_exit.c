/* 미너비니 청산 walk-forward. 6채널 경로(O,H,L,C,ma50,volmult).
 * 하드%손절(<=10%) + 본전규칙 + 50일선 트레일(수익 길게) + 반절 + violations(3연속저점).
 * argv[1]=dir argv[2]=first_test_year. 종가진입. 최대손실 추적(<=10% 검증). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define COST 0.0035
static int N,MAXH,NCH; static int *yr; static double *ea,*path;
static void load(const char*d){char p[512];FILE*f;
 snprintf(p,512,"%s/meta.txt",d);f=fopen(p,"r");if(!f){fprintf(stderr,"no dir\n");exit(1);}if(fscanf(f,"%d %d %d",&N,&MAXH,&NCH)!=3)exit(1);fclose(f);
 yr=malloc(4*N);ea=malloc(16*N);path=malloc(8*(size_t)N*MAXH*NCH);
 snprintf(p,512,"%s/year.i32",d);f=fopen(p,"rb");if(fread(yr,4,N,f)!=(size_t)N){}fclose(f);
 snprintf(p,512,"%s/ea.f64",d);f=fopen(p,"rb");if(fread(ea,8,2*N,f)!=(size_t)2*N){}fclose(f);
 snprintf(p,512,"%s/path.f64",d);f=fopen(p,"rb");if(fread(path,8,(size_t)N*MAXH*NCH,f)!=(size_t)N*MAXH*NCH){}fclose(f);}
/* hard=손절%, be=본전트리거%(0없음), sh=반절목표%(0없음), ma=50일트레일(1), viol=연속저점청산(1) */
static double sim(int i,double hard,double be,double sh,int ma,int viol,int*v){
 double E=ea[2*i];if(!isfinite(E)||E<=0){*v=0;return 0;}
 double stop=E*(1-hard),realized=0,ret=0,cl=NAN;int half=0,ex=0,lower=0;double pc=E;
 size_t b0=(size_t)i*MAXH*NCH;
 for(int j=0;j<MAXH;j++){size_t b=b0+(size_t)j*NCH;
  double o=path[b],h=path[b+1],l=path[b+2],c=path[b+3],m=path[b+4];
  if(!isfinite(c))continue;
  if(isfinite(o)&&o<=stop){ret=realized+(half?0.5:1.0)*(o/E-1);ex=1;break;}
  if(isfinite(l)&&l<=stop){ret=realized+(half?0.5:1.0)*(stop/E-1);ex=1;break;}
  if(sh>0&&!half&&isfinite(h)&&h>=E*(1+sh)){realized=0.5*sh;half=1;if(stop<E)stop=E;}
  if(be>0&&stop<E&&isfinite(h)&&h>=E*(1+be))stop=E;                    /* 본전 이상 규칙 */
  if(ma&&isfinite(m)&&isfinite(c)&&c<m&&c>E*(1-hard)){ret=realized+(half?0.5:1.0)*(c/E-1);ex=1;break;} /* 50일선 이탈=추세붕괴 */
  if(viol){ if(isfinite(c)&&c<pc)lower++; else lower=0; if(lower>=3){ret=realized+(half?0.5:1.0)*(c/E-1);ex=1;break;} }
  if(isfinite(c))pc=c;
 }
 if(!ex){ /* 시간청산 */
  for(int j=MAXH-1;j>=0;j--){double c=path[b0+(size_t)j*NCH+3];if(isfinite(c)){cl=c;break;}}
  if(!isfinite(cl)){*v=0;return 0;} ret=realized+(half?0.5:1.0)*(cl/E-1);
 }
 *v=1;return ret-2*COST;}
static void stat(double hd,double be,double sh,int ma,int vi,int TY,int test,double*mn,double*po,double*t,int*n,double*hold,double*worst){
 double s=0,sq=0,sw=0,sl=0,shd=0,wr=0;int nn=0,nw=0,nl=0;
 for(int i=0;i<N;i++){int in=test?(yr[i]==TY):(yr[i]<TY);if(!in)continue;int v;double r=sim(i,hd,be,sh,ma,vi,&v);if(!v)continue;
  s+=r;sq+=r*r;nn++;if(r>0){sw+=r;nw++;}else if(r<0){sl+=r;nl++;} if(r<wr)wr=r; }
 *n=nn;*mn=nn?s/nn:0;double var=nn?sq/nn-(*mn)*(*mn):0,sd=var>0?sqrt(var):0;
 *t=(nn&&sd>0)?(*mn)/sd*sqrt((double)nn):0;*po=(nw&&nl)?(sw/nw)/(-(sl/nl)):NAN;*hold=0;*worst=wr;}
int main(int argc,char**argv){load(argv[1]);int y0=atoi(argv[2]);
 double H[]={0.05,0.08,0.10},B[]={0,0.05,0.10},SH[]={0,0.15,0.25};int MA[]={0,1},VI[]={0,1};
 printf("== 미너비니 청산 walk-forward (하드손절<=10%%) ==\n");
 int pos=0,tot=0;double msum=0;double gworst=0;char bl[80]="";
 for(int TY=y0;TY<=2026;TY++){
  double bh=0,bb=0,bs=0;int bma=0,bvi=0;double btt=-1e9;int found=0;
  for(int a=0;a<3;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++)for(int d=0;d<2;d++)for(int e=0;e<2;e++){
   double m,po,t,hd_,wr;int n;stat(H[a],B[b],SH[c],MA[d],VI[e],TY,0,&m,&po,&t,&n,&hd_,&wr);
   if(n<100)continue;if(m>0&&t>btt){btt=t;bh=H[a];bb=B[b];bs=SH[c];bma=MA[d];bvi=VI[e];found=1;}}
  if(!found){printf("  %d:(train 양수無)\n",TY);continue;}
  double m,po,t,hd_,wr;int n;stat(bh,bb,bs,bma,bvi,TY,1,&m,&po,&t,&n,&hd_,&wr);
  if(n<20)continue;
  printf("  %d: 손절%.0f%%/본전%.0f%%/반절%.0f%%/50MA%d/viol%d | n=%d 평균%+.2f%% 손익비%.2f t%+.1f 최악%.0f%%\n",
   TY,bh*100,bb*100,bs*100,bma,bvi,n,m*100,po,t,wr*100);
  tot++;if(m>0)pos++;msum+=m*100; if(wr<gworst)gworst=wr;
 }
 printf("  >> 양수 %d/%d, 평균 %+.2f%%/트레이드, 전기간 최악손실 %.0f%%\n",pos,tot,tot?msum/tot:0,gworst*100);
 return 0;}
