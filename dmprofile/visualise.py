    max_param = 4

    with open(likelihood.out_dir + '/final_'+str(comp)+'.dat', "r") as f:
        reader = csv.reader(f, delimiter=';')
        data = [row for row in reader]
        
    fcn_list = [d[1] for d in data]
    params = np.array([d[-4:] for d in data], dtype=float)
    DL = np.array([d[2] for d in data], dtype=float)
    DL_min = np.amin(DL[np.isfinite(DL)])
    print('MIN', DL_min)
    alpha = DL_min - DL
    alpha = np.exp(alpha)
    m = (alpha > vmin)
    fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
    params = params[m,:]
    alpha = alpha[m]

    fig  = plt.figure(figsize=(7,5))
    ax1  = fig.add_axes([0.10,0.10,0.70,0.85])
    cmap = cm.hot_r
    norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)

    for i in range(min(len(fcn_list),50)):

        fcn_i = fcn_list[i].replace('\'', '')
        
        k = simplifier.count_params([fcn_i], max_param)[0]
        measured = params[i,:k]

        print('%i of %i:'%(i+1,len(fcn_list)), fcn_i)
        
        try:
            fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
            if k == 0:
                eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
            elif k==1:
                eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
            elif k==2:
                eq_numpy = sympy.lambdify([x, a0, a1], eq, modules=["numpy"])
            elif k==3:
                eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, modules=["numpy"])
            elif k==4:
                eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, modules=["numpy"])
            ypred = likelihood.get_pred(likelihood.xvar, measured, eq_numpy, integrated=integrated)
        except:
            if try_integration:
                fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                if k == 0:
                    eq_numpy = sympy.lambdify([x], eq, modules=["numpy"])
                elif k==1:
                    eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                elif k==2:
                    eq_numpy = sympy.lambdify([x, a0, a1], eq, modules=["numpy"])
                elif k==3:
                    eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, modules=["numpy"])
                elif k==4:
                    eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, modules=["numpy"])
                ypred = likelihood.get_pred(likelihood.xvar, measured, eq_numpy, integrated=integrated)
            else:
                continue