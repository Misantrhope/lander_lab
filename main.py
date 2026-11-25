from __future__ import annotations
import argparse, csv, math, os, threading
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Rectangle

BODIES = {
 'Луна':(1.62,'Спутник Земли','Почти отсутствует атмосфера; реголит.'),
 'Меркурий':(3.70,'Планета','Практически без атмосферы; кратеры.'),
 'Венера':(8.87,'Планета','Очень плотная атмосфера; в модели не учитывается.'),
 'Земля':(9.81,'Планета','Плотная атмосфера; в модели не учитывается.'),
 'Марс':(3.71,'Планета','Разреженная атмосфера; в модели не учитывается.'),
 'Церера':(.27,'Карликовая планета','Слабая гравитация; кратерная поверхность.'),
 'Ио':(1.80,'Спутник Юпитера','Вулканическая поверхность; упрощённый рельеф.'),
 'Европа':(1.31,'Спутник Юпитера','Ледяная поверхность; упрощённый рельеф.'),
 'Ганимед':(1.43,'Спутник Юпитера','Ледяно-каменистая поверхность.'),
 'Каллисто':(1.24,'Спутник Юпитера','Сильно кратерированная поверхность.'),
 'Титан':(1.35,'Спутник Сатурна','Плотная атмосфера; в модели не учитывается.'),
 'Энцелад':(.113,'Спутник Сатурна','Ледяная поверхность; очень слабая гравитация.'),
 'Тритон':(.779,'Спутник Нептуна','Ледяная поверхность.'),
 'Плутон':(.62,'Карликовая планета','Ледяная поверхность; слабая гравитация.')}
TERRAINS=['Пологий','Волнистый','Холмистый','Гористый','Кратерный']
G0=9.80665; VE=2200.; DT=.12

@dataclass(frozen=True)
class Design:
    dry_mass:float
    fuel_mass:float
    thrust:float
    support:int

@dataclass
class Settings:
    body:str='Луна'
    iterations:int=90
    seed:int=42
    output:str='results'
    no_animation:bool=False
    no_show:bool=False
    fps:int=12
    max_time:float=90.
    episodes_per_design:int=3


def terrain(x,kind):
    x=np.asarray(x)
    if kind==0:return .10*np.sin(.6*x)+.06*np.sin(1.4*x)
    if kind==1:return .28*np.sin(.43*x)+.13*np.cos(.95*x)
    if kind==2:return .24*np.sin(.52*x)+.42*np.exp(-((x+3)/1.6)**2)
    if kind==3:return .32*np.sin(.4*x)+1.3*np.exp(-((x+4.8)/1.8)**2)+1.0*np.exp(-((x-4.5)/1.6)**2)
    return -.65*np.exp(-(x/2.4)**4)+.55*np.exp(-((np.abs(x)-3.0)/.8)**2)

def slope(x,k):return float((terrain(x+.03,k)-terrain(x-.03,k))/.06)

def contact_metrics(d:Design,x:float,vx:float,vy:float,k:int):

    mass=d.dry_mass + (65 if d.support==0 else 12*d.support)
    n=d.support
    radius=0.95+(.045*(n-3) if n else .18)
    s=slope(x,k)
    left=float(terrain(x-radius,k));right=float(terrain(x+radius,k))
    dh=abs(right-left)
    if n:
        stroke=.16+.008*(n-3)

        tip=dh/(2*radius)+.05*abs(vx)
        stable=tip<.34 and abs(s)<.70
        # параллельные пружины: нагрузка на опору обратно пропорциональна n
        force_per_leg=mass*(vy*vy)/(2*max(stroke,.01)*n)
        peak_decel=vy*vy/(2*stroke)
        overload=force_per_leg>1700 or peak_decel>14
    else:
        stroke=.55
        tip=dh/(2*radius)+.025*abs(vx)
        stable=tip<.50 and abs(s)<1.0
        force_per_leg=0
        peak_decel=vy*vy/(2*stroke)
        overload=peak_decel>9
    return dict(stable=stable,overload=overload,peak_decel=peak_decel,force_per_leg=force_per_leg,
                tip=tip,slope=s,stroke=stroke,radius=radius)

def run_landing(d:Design,g:float,kind:int,seed:int,max_time:float=90.,record=False):
    rng=np.random.default_rng(seed)
    x=float(rng.uniform(-2.8,2.8));vx=float(rng.normal(0,.12))
    y=float(terrain(x,kind)+rng.uniform(10.5,13.5));vy=float(rng.uniform(-.7,-.15))
    fuel=d.fuel_mass;history=[];end='тайм-аут';last=None
    steps=int(max_time/DT)
    for step in range(steps):
        h=y-float(terrain(x,kind));mass=d.dry_mass+(65 if d.support==0 else 12*d.support)+fuel
        # Гравитационно-компенсированный PD-контроллер с непрерывной тягой.
        desired_vy=-min(1.15,max(.14,.17*max(h,0)))
        ay_cmd=np.clip(1.9*(desired_vy-vy),-2.8,2.8)
        main=float(np.clip(mass*(g+ay_cmd),0,d.thrust)) if fuel>0 else 0.
        ax_cmd=np.clip(-.8*x-1.35*vx,-1.6,1.6)
        lateral=float(np.clip(mass*abs(ax_cmd),0,.25*d.thrust)) if fuel>0 else 0.
        request=main+lateral
        burn=min(fuel,request*DT/VE)
        scale=burn*VE/(request*DT) if request>0 else 0
        # слабые стохастические ускорения, одинаковая статистика при фиксированном seed
        ax=(math.copysign(lateral,ax_cmd)*scale/mass if mass else 0)+rng.normal(0,.012)
        ay=main*scale/mass-g+rng.normal(0,.012)
        vx+=ax*DT;vy+=ay*DT;x+=vx*DT;y+=vy*DT;fuel-=burn
        if record or step%8==0:history.append((step*DT,x,y,vx,vy,fuel,main/d.thrust))
        if y-float(terrain(x,kind))<=.22:
            last=contact_metrics(d,x,vx,vy,kind)
            soft=abs(vy)<=.55 and abs(vx)<=.38
            accurate=abs(x)<=.95
            success=soft and accurate and last['stable'] and not last['overload']
            end='идеальная посадка' if success else 'неудачная посадка'
            break
        if abs(x)>12 or y>28:
            end='выход за границы';break
        if fuel<=0 and h>3 and vy< -3:
            # Продолжаем падение до контакта; не подменяем неудачу тайм-аутом.
            pass
    else:step=steps-1
    if last is None:last=contact_metrics(d,x,vx,vy,kind)
    success=end=='идеальная посадка'
    return dict(success=success,reason=end,kind=kind,steps=step+1,time=(step+1)*DT,
                x=x,vx=vx,vy=vy,fuel=fuel,peak_decel=last['peak_decel'],tip=last['tip'],
                stable=last['stable'],overload=last['overload'],history=history,design=d)

def design_from_vector(v):
    return Design(float(np.clip(v[0],180,900)),float(np.clip(v[1],55,650)),float(np.clip(v[2],1200,18000)),int(v[3]))

def sample_design(rng,g):
    dry=float(rng.uniform(230,700));fuel=float(rng.uniform(90,380))
    # Размер тяги обучается в зависимости от веса, но не фиксируется заранее.
    ratio=float(rng.uniform(.8,3.4))
    thrust=float(np.clip((dry+fuel)*g*ratio,1200,18000))
    return Design(dry,fuel,thrust,int(rng.choice([0,3,4,5,6,7,8,9,10])))

def mutate(d,rng):
    return design_from_vector([d.dry_mass*rng.uniform(.80,1.20),d.fuel_mass*rng.uniform(.76,1.25),
         d.thrust*rng.uniform(.72,1.30),int(rng.choice([d.support,0,3,4,5,6,7,8,9,10]))])

def score_design(d,g,seeds,max_time):
    trials=[run_landing(d,g,i%5,int(s),max_time) for i,s in enumerate(seeds)]
    n=sum(r['success'] for r in trials)
    # Надёжность приоритетнее минимизации массы; стоимость конструкции вторична.
    score=100*n/len(trials)-.008*(d.dry_mass+d.fuel_mass+(65 if d.support==0 else 12*d.support))-.00018*d.thrust
    score-=sum(min(3,abs(r['x']))+min(3,abs(r['vy'])) for r in trials)/len(trials)
    return score,trials

def optimize(cfg:Settings):
    rng=np.random.default_rng(cfg.seed);g=BODIES[cfg.body][0]
    out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    seeds=np.arange(cfg.episodes_per_design,dtype=int)*919+cfg.seed+100
    best=None;bestscore=-1e20;rows=[];attempts=[];first_success=None
    for i in range(cfg.iterations):
        d=sample_design(rng,g) if best is None or rng.random()<.30 else mutate(best,rng)
        score,trials=score_design(d,g,seeds,cfg.max_time)
        # Финальная проверка
        if score>bestscore:bestscore=score;best=d
        success=sum(t['success'] for t in trials)
        rows.append([i+1,d.dry_mass,d.fuel_mass,d.thrust,d.support,score,success,len(trials),bestscore])
        for j,t in enumerate(trials):attempts.append([i+1,j+1,TERRAINS[t['kind']],int(t['success']),t['reason'],t['x'],t['vx'],t['vy'],t['fuel'],t['peak_decel'],t['tip']])
        if first_success is None and success==len(trials):first_success=i+1
        if (i+1)%10==0 or i==0:print(f'Поиск {i+1}/{cfg.iterations}: успешных {success}/{len(trials)}, лучший балл {bestscore:.2f}',flush=True)
    # Дополнительный поиск до идеального результата на всех пяти рельефах с фиксированным набором seeds.
    check_seeds=[cfg.seed+5000+i*71 for i in range(5)]
    candidates=sorted(rows,key=lambda r:r[5],reverse=True)[:12]
    finalists=[]
    for row in candidates:
        d=Design(row[1],row[2],row[3],int(row[4]))
        tests=[run_landing(d,g,k,check_seeds[k],cfg.max_time) for k in range(5)]
        finalists.append((sum(t['success'] for t in tests),sum(abs(t['vy'])+abs(t['x']) for t in tests),d,tests))
    finalists.sort(key=lambda v:(-v[0],v[1]))
    nbest,_,best,checks=finalists[0]

    anim=[]
    for k in range(2):
        candidates_bad=[(r,dd) for r,dd in [(run_landing(Design(float(a[1]),float(a[2]),float(a[3]),int(a[4])),g,k,cfg.seed+8000+k,cfg.max_time),a) for a in rows[:min(12,len(rows))]] if not r['success']]
        if candidates_bad:
            bad_d=Design(float(candidates_bad[0][1][1]),float(candidates_bad[0][1][2]),float(candidates_bad[0][1][3]),int(candidates_bad[0][1][4]))
            anim.append(run_landing(bad_d,g,k,cfg.seed+8000+k,cfg.max_time,True))
    for k in range(5):anim.append(run_landing(best,g,k,check_seeds[k],cfg.max_time,True))
    save_csv(out/'design_search.csv',['итерация','сухая_масса_кг','топливо_кг','макс_тяга_Н','опоры_0_подушка','балл','успешных','испытаний','лучший_балл'],rows)
    save_csv(out/'attempts.csv',['итерация','испытание','рельеф','успех','исход','x_м','vx_м_с','vy_м_с','топливо_кг','замедление_м_с2','показатель_опрокидывания'],attempts)
    save_csv(out/'best_design.csv',['тело','g_м_с2','сухая_масса_кг','топливо_кг','тяга_Н','опоры_0_подушка','успехов_из_5'],[[cfg.body,g,best.dry_mass,best.fuel_mass,best.thrust,best.support,nbest]])
    plot_results(out,rows,attempts,best,checks,cfg.body)
    if not cfg.no_animation:
        animate(out,anim,cfg.fps,cfg.no_show)
    if not cfg.no_show:
        show_graphs_sequentially(out)
    print(f'Итог: {cfg.body}, контроль {nbest}/5, сухая масса {best.dry_mass:.0f} кг, топливо {best.fuel_mass:.0f} кг, тяга {best.thrust:.0f} Н, '+('подушка' if best.support==0 else f'{best.support} опор'))
    if nbest<5:print('ВНИМАНИЕ: идеальная посадка на всех пяти контрольных рельефах НЕ достигнута; поиск завершён по лимиту.')
    return best,nbest

def save_csv(path,header,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def plot_results(out,rows,attempts,best,checks,body):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120})
    it=np.array([r[0] for r in rows]);score=np.array([r[5] for r in rows]);bestscore=np.array([r[8] for r in rows])
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(it,score,alpha=.35,label='Оценка кандидата');ax.plot(it,bestscore,lw=2,label='Лучшая оценка');ax.set(xlabel='Итерация оптимизации',ylabel='Оценка конструкции',title=f'Подбор параметров: {body}');ax.legend();fig.tight_layout();fig.savefig(out/'optimization.png');plt.close(fig)
    a=np.asarray(attempts,dtype=object);s=np.array([int(r[3]) for r in attempts]);
    ok=np.cumsum(s);fail=np.cumsum(1-s)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(np.arange(len(s))+1,ok,label='Удачные');ax.plot(np.arange(len(s))+1,fail,label='Неудачные');ax.set(xlabel='Количество испытаний',ylabel='Накопленное число посадок',title='Результаты в ходе поиска');ax.legend();fig.tight_layout();fig.savefig(out/'success_failure.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));xs=np.arange(len(rows));ax.plot(xs,[r[1] for r in rows],label='Сухая масса, кг');ax.plot(xs,[r[2] for r in rows],label='Топливо, кг');ax.plot(xs,[r[3]/10 for r in rows],label='Тяга, Н / 10');ax.set(xlabel='Итерация',ylabel='Значение (масштаб тяги ×0,1)',title='Эволюция параметров модуля');ax.legend();fig.tight_layout();fig.savefig(out/'design_evolution.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));xx=np.linspace(-9,9,400)
    for k in range(5):ax.plot(xx,terrain(xx,k),label=TERRAINS[k])
    ax.set(xlabel='Координата, м',ylabel='Высота поверхности, м',title='Пять профилей поверхности');ax.legend(ncol=3);fig.tight_layout();fig.savefig(out/'terrain.png');plt.close(fig)

def animate(out,episodes,fps,no_show):
    if not episodes:return

    fig,ax=plt.subplots(figsize=(9,5.5));frames=[]
    for j,r in enumerate(episodes):
        hist=r['history'];stride=max(5,int(round(1/(DT*fps))))
        frames.extend((j,i) for i in range(0,len(hist),stride))
        if frames[-1]!=(j,len(hist)-1):frames.append((j,len(hist)-1))
    def update(frame):
        j,i=frame;r=episodes[j];d=r['design'];hist=r['history'];t,x,y,vx,vy,fuel,throttle=hist[i]
        ax.clear();xx=np.linspace(-11,11,300);yy=terrain(xx,r['kind']);ax.fill_between(xx,yy,-3,color='#aab4c3');ax.plot(xx,yy,color='#596477')
        hh=np.array(hist[:i+1]);ax.plot(hh[:,1],hh[:,2],alpha=.45,color='#2378b4')
        col='#239c71' if r['success'] else '#d75d4a'
        ax.add_patch(Rectangle((x-.35,y-.24),.7,.48,facecolor='#4d6682',edgecolor='white'))
        if d.support:

            for shift in (-.35,.35):ax.plot([x+shift,x+shift*2],[y-.18,y-.65],color='#333',lw=2)
        else:ax.add_patch(Polygon([[x-.55,y-.28],[x+.55,y-.28],[x+.65,y-.55],[x-.65,y-.55]],facecolor='#e2a64a'))
        if throttle>.02:ax.plot([x,x],[y-.24,y-.24-.5*throttle],color='#e87a36',lw=4)
        ax.set(xlim=(-11,11),ylim=(-3,19),xlabel='Горизонтальная координата, м',ylabel='Высота, м',title=f'Попытка {j+1}/{len(episodes)} — {TERRAINS[r["kind"]]} — {"УДАЧНАЯ" if r["success"] else "НЕУДАЧНАЯ"}')
        ax.text(.02,.97,f'Время {t:.1f} с | vy {vy:.2f} м/с | vx {vx:.2f} м/с | топливо {fuel:.1f} кг\nТяга {d.thrust:.0f} Н | масса {d.dry_mass:.0f} кг | '+('подушка' if d.support==0 else f'{d.support} опор'),transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        if i==len(hist)-1:ax.text(.5,.45,r['reason'].upper(),color=col,fontsize=17,fontweight='bold',ha='center',transform=ax.transAxes)
        return []
    anim=FuncAnimation(fig,update,frames=frames,interval=1000/fps,repeat=False,blit=False)
    anim.save(out/'landing_animation.gif',writer=PillowWriter(fps=fps),dpi=62)
    update(frames[-1]);fig.savefig(out/'animation_final.png');plt.close(fig)
    if not no_show:

        try:
            from PIL import Image,ImageTk,ImageSequence
            import tkinter as tk
            root=tk.Tk();root.title('Полная анимация попыток посадки')
            gif=Image.open(out/'landing_animation.gif');canvas=tk.Label(root);canvas.pack()
            cache={};total=getattr(gif,'n_frames',1)
            state={'i':0,'pause':False,'closed':False,'after_id':None}
            def close_animation():
                state['closed']=True
                job=state['after_id']
                state['after_id']=None
                if job is not None:
                    try:root.after_cancel(job)
                    except tk.TclError:pass
                try:root.destroy()
                except tk.TclError:pass
            def tick():
                state['after_id']=None
                if state['closed'] or not root.winfo_exists():return
                if not state['pause']:
                    idx=state['i'];gif.seek(idx)
                    img=ImageTk.PhotoImage(gif.convert('RGB'))
                    cache['image']=img;canvas.configure(image=img)
                    state['i']=(idx+1)%total
                if not state['closed']:
                    state['after_id']=root.after(max(1,int(1000/fps)),tick)
            root.protocol('WM_DELETE_WINDOW',close_animation)
            tk.Button(root,text='Пауза / продолжить',command=lambda:state.update(pause=not state['pause'])).pack()
            tick();root.mainloop()
        except Exception as exc:print('GIF сохранён; графическое окно недоступно:',exc)

def planet_image(body,size=320):
    """Более детальная процедурная иллюстрация: характерные цвета и узоры, не фотоснимок."""
    from PIL import Image, ImageDraw, ImageFilter
    seed=list(BODIES).index(body)+101
    rng=np.random.default_rng(seed)
    n=size

    def noise(scales):
        result=np.zeros((n,n),dtype=float)
        for cells,weight in scales:
            small=rng.integers(0,256,(cells,cells),dtype=np.uint8)
            layer=Image.fromarray(small,'L').resize((n,n),Image.Resampling.BICUBIC)
            result+=(np.asarray(layer,dtype=float)/255-.5)*weight
        return result
    rough=noise([(5,1),(12,.5),(30,.22),(75,.10)])
    fine=noise([(13,1),(38,.4),(90,.2)])
    yy,xx=np.mgrid[:n,:n];cx=cy=n/2;R=n*.43
    dx=(xx-cx)/R;dy=(yy-cy)/R;r2=dx**2+dy**2;inside=r2<=1
    z=np.sqrt(np.maximum(0,1-r2));lat=np.arcsin(np.clip(dy,-1,1));lon=np.arctan2(dx,z+.001)
    bases={
      'Луна':(164,163,160),'Меркурий':(151,139,125),'Венера':(224,181,111),
      'Земля':(30,91,164),'Марс':(181,91,59),'Церера':(138,135,129),
      'Ио':(223,188,100),'Европа':(205,196,173),'Ганимед':(142,128,113),
      'Каллисто':(102,97,91),'Титан':(202,148,87),'Энцелад':(208,225,235),
      'Тритон':(178,183,180),'Плутон':(179,150,132)}
    rgb=np.zeros((n,n,3),dtype=float)
    rgb[:]=bases[body]
    if body=='Земля':
        continents=(np.sin(3.2*lon+2*np.sin(2*lat))+ .7*np.sin(7*lat-2*lon)+.65*rough)>.48
        land=np.where(((np.sin(lat*3)+rough)>.1)[:,:,None], np.array([67,118,66]),np.array([163,141,94]))
        rgb=np.where(continents[:,:,None],land,rgb)
        ice=np.abs(lat)>1.12
        rgb=np.where(ice[:,:,None],[226,236,240],rgb)
        cloud=(fine+.32*np.sin(14*lon+8*lat))>.43
        rgb=np.where(cloud[:,:,None],rgb*.38+np.array([240,246,247])*.62,rgb)
    elif body in ('Венера','Титан'):
        band=np.sin(18*lat+2*rough+1.2*np.sin(3*lon))
        rgb+=band[:,:,None]*np.array([14,12,9])
        rgb+=fine[:,:,None]*12
        if body=='Венера':rgb+=np.cos(9*lat+3*lon)[:,:,None]*np.array([10,9,7])
    elif body=='Марс':
        dark=(rough+.25*np.sin(5*lon+lat*3))>.15
        rgb=np.where(dark[:,:,None],rgb*.69,rgb)
        polar=np.abs(lat)>1.24
        rgb=np.where(polar[:,:,None],[220,202,178],rgb)
    elif body=='Ио':
        sulfur=(rough+.3*np.cos(9*lon+5*lat))>.15
        rgb=np.where(sulfur[:,:,None],[239,207,101],[173,137,72]).astype(float)
        rgb+=fine[:,:,None]*20
    elif body in ('Европа','Энцелад','Тритон'):
        rgb+=rough[:,:,None]*np.array([17,19,22])
        cracks=np.sin(24*lon+8*np.sin(6*lat)+3*rough)
        rgb=np.where((cracks>.965)[:,:,None],rgb*.65,rgb)
    elif body=='Плутон':
        heart=np.exp(-((lon-.15)/.54)**2-((lat+.18)/.39)**2)
        rgb=rgb*(1-heart[:,:,None]*.55)+np.array([226,215,196])*heart[:,:,None]*.55
        rgb+=rough[:,:,None]*23
    else:
        rgb+=rough[:,:,None]*27+fine[:,:,None]*10

    if body in ('Луна','Меркурий','Церера','Ганимед','Каллисто'):
        crater=np.zeros((n,n),dtype=float)
        for _ in range(120 if body in ('Луна','Меркурий','Каллисто') else 70):
            u,v=rng.uniform(-.9,.9,2);rad=rng.uniform(.008,.065)
            dist=np.sqrt((dx-u)**2+(dy-v)**2)
            crater-=np.exp(-(dist/(rad*.65))**4)*rng.uniform(4,15)
            crater+=np.exp(-((dist-rad)/(.18*rad+.004))**2)*rng.uniform(4,13)
        rgb+=crater[:,:,None]
    light=np.clip(.21+.83*(.52*dx-.30*dy+.82*z),.17,1.12)
    rgb=np.clip(rgb*light[:,:,None],0,255).astype(np.uint8)
    background=np.zeros((n,n,3),dtype=np.uint8);background[:]=[12,24,44]
    background[inside]=rgb[inside]
    im=Image.fromarray(background,'RGB')
    return im

def choose_body_gui():
    import tkinter as tk
    from PIL import ImageTk
    root=tk.Tk();root.title('Выбор небесного тела');root.geometry('1000x750');root.minsize(900,700);root.configure(bg='#0c182c')
    chosen={'body':None};left=tk.Frame(root,bg='#15263d');left.pack(side='left',fill='y',padx=16,pady=16)
    tk.Label(left,text='ВЫБЕРИТЕ НЕБЕСНОЕ ТЕЛО',fg='#e6edf6',bg='#15263d',font=('Segoe UI',13,'bold')).pack(padx=12,pady=14)
    lb=tk.Listbox(left,font=('Segoe UI',12),width=20,height=19,bg='#1d334f',fg='white',selectbackground='#2b85a6',relief='flat',exportselection=False)
    lb.pack(padx=12,pady=(0,14),fill='both',expand=True)
    for b in BODIES:lb.insert('end',b)
    right=tk.Frame(root,bg='#0c182c');right.pack(side='left',fill='both',expand=True,padx=18,pady=16)
    title=tk.Label(right,text='',fg='white',bg='#0c182c',font=('Segoe UI',22,'bold'));title.pack(anchor='w')
    image=tk.Label(right,bg='#0c182c',width=320,height=320);image.pack(pady=10)
    detail=tk.Label(right,text='',fg='#c4d4e8',bg='#0c182c',font=('Segoe UI',11),justify='left',wraplength=420);detail.pack(anchor='w',pady=8)
    cache={};requests={'latest':None};pending=set();closed={'value':False}
    def display(b):
        if b not in cache or closed['value']:return
        photo=ImageTk.PhotoImage(cache[b]);image.configure(image=photo);image.image=photo
    def load_async(b):
        if b in pending or b in cache:return
        pending.add(b)
        def worker():
            try:pic=planet_image(b)
            except Exception:pic=None
            def done():
                pending.discard(b)
                if closed['value']:return
                if pic is not None:
                    cache[b]=pic
                    if requests['latest']==b:display(b)
            try:root.after(0,done)
            except (RuntimeError,tk.TclError):pass
        threading.Thread(target=worker,daemon=True).start()
    def refresh(_=None):
        if not lb.curselection():return
        b=lb.get(lb.curselection()[0]);requests['latest']=b
        g,typ,desc=BODIES[b];title.configure(text=b)
        detail.configure(text=f'{typ}\nУскорение свободного падения: {g:.3f} м/с²\n{desc}')
        if b in cache:display(b)
        else:
            image.configure(image='',text='Создание изображения…',fg='#829ab6',font=('Segoe UI',11))
            image.image=None
            load_async(b)
    def start():
        if lb.curselection():chosen['body']=lb.get(lb.curselection()[0]);closed['value']=True;root.destroy()
    def close():closed['value']=True;root.destroy()
    root.protocol('WM_DELETE_WINDOW',close)
    tk.Button(right,text='СТАРТ ▶️',font=('Segoe UI',11,'bold'),bg='#2b85a6',fg='white',activebackground='#45a6c2',command=start,padx=18,pady=10).pack(pady=24,fill='x')
    lb.selection_set(0);refresh();lb.bind('<<ListboxSelect>>',refresh);root.mainloop();return chosen['body']

def show_graphs_sequentially(out):

    try:
        import tkinter as tk
        from PIL import Image,ImageTk
        names=[('optimization.png','1. Оптимизация конструкции'),
               ('success_failure.png','2. Удачные и неудачные посадки'),
               ('design_evolution.png','3. Изменение массы, топлива и тяги'),
               ('terrain.png','4. Профили рельефа')]
        names=[(out/f,t) for f,t in names if (out/f).exists()]
        if not names:return
        root=tk.Tk();root.title('Результаты эксперимента — графики');root.geometry('1000x700')
        root.configure(bg='#0c182c');state={'i':0};photos={}
        title=tk.Label(root,bg='#0c182c',fg='white',font=('Segoe UI',15,'bold'));title.pack(pady=12)
        panel=tk.Label(root,bg='#0c182c');panel.pack(fill='both',expand=True,padx=14,pady=8)
        controls=tk.Frame(root,bg='#0c182c');controls.pack(pady=12)
        def render():
            i=state['i'];path,heading=names[i];title.configure(text=f'{heading}  ({i+1}/{len(names)})')
            if i not in photos:
                pic=Image.open(path).convert('RGB');pic.thumbnail((950,570),Image.Resampling.LANCZOS)
                photos[i]=ImageTk.PhotoImage(pic)
            panel.configure(image=photos[i]);panel.image=photos[i]
            prev.configure(state='normal' if i>0 else 'disabled')
            next_.configure(text='Следующий график →' if i<len(names)-1 else 'Завершить',state='normal')
        def move(delta):
            if delta>0 and state['i']==len(names)-1:root.destroy();return
            state['i']=max(0,min(len(names)-1,state['i']+delta));render()
        prev=tk.Button(controls,text='← Предыдущий',command=lambda:move(-1),font=('Segoe UI',11),padx=12)
        prev.pack(side='left',padx=8)
        next_=tk.Button(controls,text='Следующий график →',command=lambda:move(1),font=('Segoe UI',11),padx=12)
        next_.pack(side='left',padx=8)
        root.bind('<Right>',lambda _:move(1));root.bind('<Left>',lambda _:move(-1))
        render();root.mainloop()
    except Exception as exc:
        print('Графики сохранены, но окно просмотра недоступно:',exc)

def main():
    p=argparse.ArgumentParser(description='Самообучающийся посадочный модуль: поиск конструкции и контроллер посадки')
    p.add_argument('--body',choices=list(BODIES),default=None);p.add_argument('--iterations',type=int,default=90)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='results');p.add_argument('--no-gui',action='store_true')
    p.add_argument('--no-animation',action='store_true');p.add_argument('--no-show',action='store_true');p.add_argument('--fps',type=int,default=12)
    p.add_argument('--max-time',type=float,default=90);p.add_argument('--trials',type=int,default=3)
    a=p.parse_args()
    body=a.body if a.body else ('Луна' if a.no_gui else choose_body_gui())
    if body is None:return
    if a.iterations<1 or a.trials<1 or a.fps<1 or a.max_time<5:p.error('Неверные параметры расчёта')
    cfg=Settings(body=body,iterations=a.iterations,seed=a.seed,output=a.output,no_animation=a.no_animation,no_show=a.no_show,fps=a.fps,max_time=a.max_time,episodes_per_design=a.trials)
    optimize(cfg)
if __name__=='__main__':main()
