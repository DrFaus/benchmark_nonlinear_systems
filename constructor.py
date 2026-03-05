tamanio_poblacion = 600
numero_generaciones = 2000
pruebas = 30
f_objetivo_critico = 1.0
factor_01 = 1 # CSTR: 0 moderadamente no lineal, 1 altamente no lineal
factor_02 = 1 # FIS Type: 0 Mamdani, 1 Tsukamoto
factor_03 = 1 # Parametrización: 0 Discreto, 1 continuo
factor_04 = 1 # 0 incremental-form, 1 level-form

import pandas as pd

if factor_01 == 0:
    df = pd.read_csv("https://raw.githubusercontent.com/DrFaus/benchmark_nonlinear_systems/refs/heads/main/datos_casos/caso_01_semilla_147.csv")
else:
    df = pd.read_csv("https://raw.githubusercontent.com/DrFaus/benchmark_nonlinear_systems/refs/heads/main/datos_casos/caso_02_semilla_103.csv")

df = df.loc[:5999,:]
x1_prom = float(df.y1.mean())
x2_prom = float(df.u.mean())

y_prom = factor_04 * float(df.y2.mean())

numero_datos_entrenamiento = df.shape[0]

list_x1 = list(df.y1)
list_x2 = list(df.u)
list_respuesta = list(df.y2)

importar_datos = f'''
        x1(1:200)    = [{",".join([str(i)+"d0" for i in list_x1[0:200]])}]
        x1(201:400)  = [{",".join([str(i)+"d0" for i in list_x1[200:400]])}]
        x1(401:600)  = [{",".join([str(i)+"d0" for i in list_x1[400:600]])}]
        x1(601:800)  = [{",".join([str(i)+"d0" for i in list_x1[600:800]])}]
        x1(801:1000) = [{",".join([str(i)+"d0" for i in list_x1[800:1000]])}]
        x1(1001:1200)= [{",".join([str(i)+"d0" for i in list_x1[1000:1200]])}]
        x1(1201:1400)= [{",".join([str(i)+"d0" for i in list_x1[1200:1400]])}]
        x1(1401:1600)= [{",".join([str(i)+"d0" for i in list_x1[1400:1600]])}]
        x1(1601:1800)= [{",".join([str(i)+"d0" for i in list_x1[1600:1800]])}]
        x1(1801:2000)= [{",".join([str(i)+"d0" for i in list_x1[1800:2000]])}]
        x1(2001:2200)= [{",".join([str(i)+"d0" for i in list_x1[2000:2200]])}]
        x1(2201:2400)= [{",".join([str(i)+"d0" for i in list_x1[2200:2400]])}]
        x1(2401:2600)= [{",".join([str(i)+"d0" for i in list_x1[2400:2600]])}]
        x1(2601:2800)= [{",".join([str(i)+"d0" for i in list_x1[2600:2800]])}]
        x1(2801:3000)= [{",".join([str(i)+"d0" for i in list_x1[2800:3000]])}]
        x1(3001:3200)= [{",".join([str(i)+"d0" for i in list_x1[3000:3200]])}]
        x1(3201:3400)= [{",".join([str(i)+"d0" for i in list_x1[3200:3400]])}]
        x1(3401:3600)= [{",".join([str(i)+"d0" for i in list_x1[3400:3600]])}]
        x1(3601:3800)= [{",".join([str(i)+"d0" for i in list_x1[3600:3800]])}]
        x1(3801:4000)= [{",".join([str(i)+"d0" for i in list_x1[3800:4000]])}]
        x1(4001:4200)= [{",".join([str(i)+"d0" for i in list_x1[4000:4200]])}]
        x1(4201:4400)= [{",".join([str(i)+"d0" for i in list_x1[4200:4400]])}]
        x1(4401:4600)= [{",".join([str(i)+"d0" for i in list_x1[4400:4600]])}]
        x1(4601:4800)= [{",".join([str(i)+"d0" for i in list_x1[4600:4800]])}]
        x1(4801:5000)= [{",".join([str(i)+"d0" for i in list_x1[4800:5000]])}]
        x1(5001:5200)= [{",".join([str(i)+"d0" for i in list_x1[5000:5200]])}]
        x1(5201:5400)= [{",".join([str(i)+"d0" for i in list_x1[5200:5400]])}]
        x1(5401:5600)= [{",".join([str(i)+"d0" for i in list_x1[5400:5600]])}]
        x1(5601:5800)= [{",".join([str(i)+"d0" for i in list_x1[5600:5800]])}]
        x1(5801:6000)= [{",".join([str(i)+"d0" for i in list_x1[5800:6000]])}]
        x2(1:200)    = [{",".join([str(i)+"d0" for i in list_x2[0:200]])}]
        x2(201:400)  = [{",".join([str(i)+"d0" for i in list_x2[200:400]])}]
        x2(401:600)  = [{",".join([str(i)+"d0" for i in list_x2[400:600]])}]
        x2(601:800)  = [{",".join([str(i)+"d0" for i in list_x2[600:800]])}]
        x2(801:1000) = [{",".join([str(i)+"d0" for i in list_x2[800:1000]])}]
        x2(1001:1200)= [{",".join([str(i)+"d0" for i in list_x2[1000:1200]])}]
        x2(1201:1400)= [{",".join([str(i)+"d0" for i in list_x2[1200:1400]])}]
        x2(1401:1600)= [{",".join([str(i)+"d0" for i in list_x2[1400:1600]])}]
        x2(1601:1800)= [{",".join([str(i)+"d0" for i in list_x2[1600:1800]])}]
        x2(1801:2000)= [{",".join([str(i)+"d0" for i in list_x2[1800:2000]])}]
        x2(2001:2200)= [{",".join([str(i)+"d0" for i in list_x2[2000:2200]])}]
        x2(2201:2400)= [{",".join([str(i)+"d0" for i in list_x2[2200:2400]])}]
        x2(2401:2600)= [{",".join([str(i)+"d0" for i in list_x2[2400:2600]])}]
        x2(2601:2800)= [{",".join([str(i)+"d0" for i in list_x2[2600:2800]])}]
        x2(2801:3000)= [{",".join([str(i)+"d0" for i in list_x2[2800:3000]])}]
        x2(3001:3200)= [{",".join([str(i)+"d0" for i in list_x2[3000:3200]])}]
        x2(3201:3400)= [{",".join([str(i)+"d0" for i in list_x2[3200:3400]])}]
        x2(3401:3600)= [{",".join([str(i)+"d0" for i in list_x2[3400:3600]])}]
        x2(3601:3800)= [{",".join([str(i)+"d0" for i in list_x2[3600:3800]])}]
        x2(3801:4000)= [{",".join([str(i)+"d0" for i in list_x2[3800:4000]])}]
        x2(4001:4200)= [{",".join([str(i)+"d0" for i in list_x2[4000:4200]])}]
        x2(4201:4400)= [{",".join([str(i)+"d0" for i in list_x2[4200:4400]])}]
        x2(4401:4600)= [{",".join([str(i)+"d0" for i in list_x2[4400:4600]])}]
        x2(4601:4800)= [{",".join([str(i)+"d0" for i in list_x2[4600:4800]])}]
        x2(4801:5000)= [{",".join([str(i)+"d0" for i in list_x2[4800:5000]])}]
        x2(5001:5200)= [{",".join([str(i)+"d0" for i in list_x2[5000:5200]])}]
        x2(5201:5400)= [{",".join([str(i)+"d0" for i in list_x2[5200:5400]])}]
        x2(5401:5600)= [{",".join([str(i)+"d0" for i in list_x2[5400:5600]])}]
        x2(5601:5800)= [{",".join([str(i)+"d0" for i in list_x2[5600:5800]])}]
        x2(5801:6000)= [{",".join([str(i)+"d0" for i in list_x2[5800:6000]])}]
        respuesta(1:200)    = [{",".join([str(i)+"d0" for i in list_respuesta[0:200]])}]
        respuesta(201:400)  = [{",".join([str(i)+"d0" for i in list_respuesta[200:400]])}]
        respuesta(401:600)  = [{",".join([str(i)+"d0" for i in list_respuesta[400:600]])}]
        respuesta(601:800)  = [{",".join([str(i)+"d0" for i in list_respuesta[600:800]])}]
        respuesta(801:1000) = [{",".join([str(i)+"d0" for i in list_respuesta[800:1000]])}]
        respuesta(1001:1200)= [{",".join([str(i)+"d0" for i in list_respuesta[1000:1200]])}]
        respuesta(1201:1400)= [{",".join([str(i)+"d0" for i in list_respuesta[1200:1400]])}]
        respuesta(1401:1600)= [{",".join([str(i)+"d0" for i in list_respuesta[1400:1600]])}]
        respuesta(1601:1800)= [{",".join([str(i)+"d0" for i in list_respuesta[1600:1800]])}]
        respuesta(1801:2000)= [{",".join([str(i)+"d0" for i in list_respuesta[1800:2000]])}]
        respuesta(2001:2200)= [{",".join([str(i)+"d0" for i in list_respuesta[2000:2200]])}]
        respuesta(2201:2400)= [{",".join([str(i)+"d0" for i in list_respuesta[2200:2400]])}]
        respuesta(2401:2600)= [{",".join([str(i)+"d0" for i in list_respuesta[2400:2600]])}]
        respuesta(2601:2800)= [{",".join([str(i)+"d0" for i in list_respuesta[2600:2800]])}]
        respuesta(2801:3000)= [{",".join([str(i)+"d0" for i in list_respuesta[2800:3000]])}]
        respuesta(3001:3200)= [{",".join([str(i)+"d0" for i in list_respuesta[3000:3200]])}]
        respuesta(3201:3400)= [{",".join([str(i)+"d0" for i in list_respuesta[3200:3400]])}]
        respuesta(3401:3600)= [{",".join([str(i)+"d0" for i in list_respuesta[3400:3600]])}]
        respuesta(3601:3800)= [{",".join([str(i)+"d0" for i in list_respuesta[3600:3800]])}]
        respuesta(3801:4000)= [{",".join([str(i)+"d0" for i in list_respuesta[3800:4000]])}]
        respuesta(4001:4200)= [{",".join([str(i)+"d0" for i in list_respuesta[4000:4200]])}]
        respuesta(4201:4400)= [{",".join([str(i)+"d0" for i in list_respuesta[4200:4400]])}]
        respuesta(4401:4600)= [{",".join([str(i)+"d0" for i in list_respuesta[4400:4600]])}]
        respuesta(4601:4800)= [{",".join([str(i)+"d0" for i in list_respuesta[4600:4800]])}]
        respuesta(4801:5000)= [{",".join([str(i)+"d0" for i in list_respuesta[4800:5000]])}]
        respuesta(5001:5200)= [{",".join([str(i)+"d0" for i in list_respuesta[5000:5200]])}]
        respuesta(5201:5400)= [{",".join([str(i)+"d0" for i in list_respuesta[5200:5400]])}]
        respuesta(5401:5600)= [{",".join([str(i)+"d0" for i in list_respuesta[5400:5600]])}]
        respuesta(5601:5800)= [{",".join([str(i)+"d0" for i in list_respuesta[5600:5800]])}]
        respuesta(5801:6000)= [{",".join([str(i)+"d0" for i in list_respuesta[5800:6000]])}]   
'''


if factor_03 == 0:
    call_rules = "call base_de_reglas(anint(x(1:9)), base)"
else:
    call_rules = "call base_de_reglas(x(1:9), base)"

funcion_base_de_reglas = '''
    subroutine base_de_reglas(reglas, base)
        double precision, dimension(9), intent(in) :: reglas
        double precision, dimension(3, 3, 3), intent(out) :: base
        integer :: fila, columna, idx

        ! reglas son los params

        ! base(entrada 1, entrada 2, salida)
        ! 1: Negativo
        ! 2: Zero
        ! 3: Positivo

        ! r1 r4 r7
        ! r2 r5 r8
        ! r3 r6 r9

        columna = 1
        fila = 1

        do idx = 1, 9
            base(fila, columna, 1) = trapezoidal([-10.0d0, -9.0d0, 1.0d0, 2.0d0], reglas(idx))
            base(fila, columna, 2) = triangular([1.0d0, 2.0d0, 3.0d0], reglas(idx))
            base(fila, columna, 3) = trapezoidal([2.0d0, 3.0d0, 9.0d0, 10.0d0], reglas(idx))

            fila = fila + 1
            if (fila > 3) then 
                fila = 1
                columna = columna + 1
            end if
        end do
    end subroutine base_de_reglas
    '''

funcion_fis_tsukamoto = '''
    double precision function fis_tsukamoto(base, params, entrada_01, entrada_02)
        double precision, dimension(28), intent(in) :: params
        double precision, dimension(3, 3, 3), intent(in) :: base
        double precision, intent(in) :: entrada_01, entrada_02
        double precision, dimension(3) :: antecedente_01, antecedente_02
        double precision :: implicacion_larsen, numerador, denominador
        double precision :: a, b, lineal_k
        integer :: i, j, k, contador

        antecedente_01(1) = trapezoidal(params(1:4), entrada_01)
        antecedente_01(2) = triangular(params(5:7), entrada_01)
        antecedente_01(3) = trapezoidal(params(8:11), entrada_01)

        antecedente_02(1) = trapezoidal(params(12:15), entrada_02)
        antecedente_02(2) = triangular(params(16:18), entrada_02)
        antecedente_02(3) = trapezoidal(params(19:22), entrada_02)

        denominador = 0.0d0
        numerador = 0.0d0

        do i = 1, 3
            do j = 1, 3
                implicacion_larsen = antecedente_01(i) * antecedente_02(j)
                denominador = denominador + implicacion_larsen 
                do k = 1, 3
                    a = params(23 + 2 * (k - 1))
                    b = params(24 + 2 * (k - 1))
                    lineal_k = lineal(a, b, implicacion_larsen)
                    numerador = numerador + implicacion_larsen * lineal_k * base(i,j,k)
                end do 
            end do 
        end do 

        denominador = merge(1.0d-10, denominador, abs(denominador) < 1.0d-10)

        fis_tsukamoto = numerador / denominador
    end function fis_tsukamoto'''

funcion_x_bar_trapecio = '''
    double precision function x_bar_trapecio(x0, x1, altura_izquierda, altura_derecha)
        double precision, intent(in) :: x0, x1, altura_izquierda, altura_derecha
        double precision :: h_i, h_d, b, x_bar

        h_i = altura_izquierda
        h_d = altura_derecha
        b = x1 - x0

        if (((h_i+h_d) == 0.0d0) .or. (b == 0.0d0)) then
            x_bar = 0.0d0
        else
            x_bar = x0 + b * (h_i + 2.0d0 * h_d) / (3.0d0 * (h_i + h_d)) 
        end if

        x_bar_trapecio = x_bar
    end function x_bar_trapecio
'''

funcion_area_trapecio = '''
    double precision function area_trapecio(x0, x1, altura_izquierda, altura_derecha)
        double precision, intent(in) :: x0, x1, altura_izquierda, altura_derecha
        double precision :: h_i, h_d, b, x_bar

        h_i = altura_izquierda
        h_d = altura_derecha
        b = x1 - x0

        area_trapecio = (h_i + h_d) * b / 2.0d0
    end function area_trapecio
'''

funcion_zona_conficto = '''
    subroutine zona_conflicto(w1, w2, h1, h2, x_bar, area_total)
        implicit none
        ! Inputs
        double precision, intent(in)  :: w1, w2, h1, h2
        ! Outputs
        double precision, intent(out) :: x_bar, area_total

        ! Locals
        integer :: areas
        double precision :: wp, w1x, w2x
        double precision :: area_1, area_2, area_3, area_4
        double precision :: x_bar_1, x_bar_2, x_bar_3, x_bar_4
        double precision :: denom

        areas = 4
        wp = 0.5d0 * (w1 + w2)

        if ((h1 + h2) < 1.0d-7) then
            x_bar = 0.0d0
            area_total = 0.0d0
            return
        end if

        ! Determine w1x, w2x and whether we split into 3 or 4 areas
        if (h1 >= 0.5d0 .and. h2 >= 0.5d0) then
            w1x = w2 + h1 * (w1 - w2)
            w2x = w1 + h2 * (w2 - w1)

        else if (h1 >= 0.5d0 .and. h2 < 0.5d0) then
            w1x = w2 + h1 * (w1 - w2)
            w2x = w2 + h2 * (w1 - w2)

        else if (h1 < 0.5d0 .and. h2 >= 0.5d0) then
            w1x = w1 + h1 * (w2 - w1)
            w2x = w1 + h2 * (w2 - w1)

        else if (h1 < 0.5d0 .and. h2 < 0.5d0 .and. h1 >= h2) then
            w1x = w2 + h1 * (w1 - w2)
            w2x = w2 + h2 * (w1 - w2)
            areas = 3

        else
            ! h1 < 0.5 and h2 < 0.5 and h1 < h2
            w1x = w1 + h1 * (w2 - w1)
            w2x = w1 + h2 * (w2 - w1)
            areas = 3
        end if

        if (areas == 3) then
            area_1 = area_trapecio(w1,  w1x, h1, h1)
            area_2 = area_trapecio(w1x, w2x, h1, h2)
            area_3 = area_trapecio(w2x, w2,  h2, h2)
            area_4 = 0.0d0

            x_bar_1 = x_bar_trapecio(w1,  w1x, h1, h1)
            x_bar_2 = x_bar_trapecio(w1x, w2x, h1, h2)
            x_bar_3 = x_bar_trapecio(w2x, w2,  h2, h2)
            x_bar_4 = 0.0d0

        else
            ! areas == 4
            area_1 = area_trapecio(w1,  w1x, h1,  h1)
            area_2 = area_trapecio(w1x, wp,  h1,  0.5d0)
            area_3 = area_trapecio(wp,  w2x, 0.5d0, h2)
            area_4 = area_trapecio(w2x, w2,  h2,  h2)

            x_bar_1 = x_bar_trapecio(w1,  w1x, h1,  h1)
            x_bar_2 = x_bar_trapecio(w1x, wp,  h1,  0.5d0)
            x_bar_3 = x_bar_trapecio(wp,  w2x, 0.5d0, h2)
            x_bar_4 = x_bar_trapecio(w2x, w2,  h2,  h2)
        end if

        area_total = area_1 + area_2 + area_3 + area_4
        denom = area_total

        if (abs(denom) < 1.0d-14) then
            x_bar = 0.0d0
            area_total = 0.0d0
            return
        end if

        x_bar = (area_1*x_bar_1 + area_2*x_bar_2 + area_3*x_bar_3 + area_4*x_bar_4) / denom

    end subroutine zona_conflicto
'''

funcion_centroide_total = '''
    subroutine centroide_total(params, fuzs, centroide)
        implicit none
        ! Inputs
        double precision, intent(in)  :: params(5)  ! [a, w1, w2, w3, b]
        double precision, intent(in)  :: fuzs(3)    ! [h1, h2, h3]
        ! Output
        double precision, intent(out) :: centroide

        ! Locals
        double precision :: a0, w1, w2, w3, b0
        double precision :: h1, h2, h3
        double precision :: areas, ponderados
        double precision :: area, ponderado
        double precision :: c, a_conf  ! c = x_bar_conflicto, a_conf = area_conflicto

        ! Unpack
        a0 = params(1)
        w1 = params(2)
        w2 = params(3)
        w3 = params(4)
        b0 = params(5)

        h1 = fuzs(1)
        h2 = fuzs(2)
        h3 = fuzs(3)

        areas = 0.0d0
        ponderados = 0.0d0

        ! Left rectangle: [a0, w1] at height h1
        area = (w1 - a0) * h1
        ponderado = area * 0.5d0 * (w1 + a0)
        areas = areas + area
        ponderados = ponderados + ponderado

        ! Conflict zone between (w1,w2) with heights (h1,h2)
        call zona_conflicto(w1, w2, h1, h2, c, a_conf)
        ponderados = ponderados + c * a_conf
        areas = areas + a_conf

        ! Conflict zone between (w2,w3) with heights (h2,h3)
        call zona_conflicto(w2, w3, h2, h3, c, a_conf)
        ponderados = ponderados + c * a_conf
        areas = areas + a_conf

        ! Right rectangle: [w3, b0] at height h3
        area = (b0 - w3) * h3
        ponderado = area * 0.5d0 * (w3 + b0)
        areas = areas + area
        ponderados = ponderados + ponderado

        ! Protect against division by zero
        if (abs(areas) < 1.0d-14) then
            centroide = 0.0d0
        else
            centroide = ponderados / areas
        end if
    end subroutine centroide_total
'''

funcion_fis_mamdani = '''
    double precision function fis_mamdani(base, params, entrada_01, entrada_02)
        double precision, dimension(34), intent(in) :: params
        double precision, dimension(3, 3, 3), intent(in) :: base
        double precision, intent(in) :: entrada_01, entrada_02
        double precision, dimension(3) :: antecedente_01, antecedente_02
        double precision :: alpha, s1, s2, s3, tau, den
        double precision :: puntos(5), fuzs(3), z
        integer :: i, j, k, contador

        antecedente_01(1) = trapezoidal(params(1:4), entrada_01)
        antecedente_01(2) = triangular(params(5:7), entrada_01)
        antecedente_01(3) = trapezoidal(params(8:11), entrada_01)

        antecedente_02(1) = trapezoidal(params(12:15), entrada_02)
        antecedente_02(2) = triangular(params(16:18), entrada_02)
        antecedente_02(3) = trapezoidal(params(19:22), entrada_02)

        s1 = 0.0d0
        s2 = 0.0d0
        s3 = 0.0d0
        tau = 8.0d0

        do i = 1, 3
            do j = 1, 3
                alpha = antecedente_01(i) * antecedente_02(j)
                s1 = s1 + base(i, j, 1) * alpha
                s2 = s2 + base(i, j, 2) * alpha
                s3 = s3 + base(i, j, 3) * alpha
            end do 
        end do 

        den = exp(tau * s1) + exp(tau * s2) + exp(tau * s3)
        fuzs(1) = exp(tau * s1) / den
        fuzs(2) = exp(tau * s2) / den
        fuzs(3) = exp(tau * s3) / den

        puntos = [params(24), params(28), params(29), params(30), params(33)]

        call centroide_total(puntos, fuzs, fis_mamdani)

    end function fis_mamdani'''

definicion_parametros_tsukamoto = f'''
        antecedente01 = {x1_prom}d0 + [-x(10), 0.0d0, x(11)]
        params(1) = -1000000.0d0
        params(2) = -100000.0d0
        params(3) = antecedente01(1)
        params(4) = antecedente01(2)

        params(5) = antecedente01(1)
        params(6) = antecedente01(2)
        params(7) = antecedente01(3)

        params(8) = antecedente01(2)
        params(9) = antecedente01(3)
        params(10) = 100000.0d0
        params(11) = 1000000.0d0

        antecedente02 = {x2_prom}d0 + [-x(12), 0.0d0, x(13)]

        params(12) = -1000000.0d0
        params(13) = -100000.0d0
        params(14) = antecedente02(1)
        params(15) = antecedente02(2)

        params(16) = antecedente02(1)
        params(17) = antecedente02(2)
        params(18) = antecedente02(3)

        params(19) = antecedente02(2)
        params(20) = antecedente02(3)
        params(21) = 100000.0d0
        params(22) = 1000000.0d0
        
        params(23) = {y_prom}d0 - x(14)
        params(24) = {y_prom}d0 - x(15)
        params(25) = {y_prom}d0
        params(26) = {y_prom}d0
        params(27) = {y_prom}d0 + x(16)
        params(28) = {y_prom}d0 + x(17)
'''

definicion_parametros_mamdani = f'''
        antecedente01 = {x1_prom}d0 + [-x(10), 0.0d0, x(11)]
        params(1) = -1000000.0d0
        params(2) = -100000.0d0
        params(3) = antecedente01(1)
        params(4) = antecedente01(2)

        params(5) = antecedente01(1)
        params(6) = antecedente01(2)
        params(7) = antecedente01(3)

        params(8) = antecedente01(2)
        params(9) = antecedente01(3)
        params(10) = 100000.0d0
        params(11) = 1000000.0d0

        antecedente02 = {x2_prom}d0 + [-x(12), 0.0d0, x(13)]

        params(12) = -1000000.0d0
        params(13) = -100000.0d0
        params(14) = antecedente02(1)
        params(15) = antecedente02(2)

        params(16) = antecedente02(1)
        params(17) = antecedente02(2)
        params(18) = antecedente02(3)

        params(19) = antecedente02(2)
        params(20) = antecedente02(3)
        params(21) = 100000.0d0
        params(22) = 1000000.0d0
        
        params(23) = {y_prom}d0 - x(14) - x(15) - 0.00001d0
        params(24) = {y_prom}d0 - x(14) - x(15)
        params(26) = {y_prom}d0 - x(14)
        params(27) = {y_prom}d0

        params(28) = {y_prom}d0 - x(14)
        params(29) = {y_prom}d0
        params(30) = {y_prom}d0 + x(16)

        params(31) = {y_prom}d0
        params(32) = {y_prom}d0 + x(16)
        params(33) = {y_prom}d0 + x(16) + x(17)
        params(34) = {y_prom}d0 + x(16) + x(17) + 0.00001d0
'''

tsukamoto_entrenamiento = f'''
    function entrenamiento(x) result(objetivo)
        double precision, dimension(:), intent(in) :: x
        double precision, dimension({numero_datos_entrenamiento}) :: x1, x2, respuesta
        double precision, dimension(3,3,3) :: base
        double precision, dimension(28) :: params
        double precision :: suma_cuadrados, prediccion_0, prediccion_1, SSE
        double precision :: RMSE, objetivo, RMSE_crit
        double precision, dimension(3) :: antecedente01, antecedente02
        integer :: i

    {importar_datos}
        
        suma_cuadrados = 0.0d0 
        RMSE_crit = 10.0
        SSE = 0.0d0
        {call_rules}

{definicion_parametros_tsukamoto}

        prediccion_0 = x1(1)
        do i = 1, {numero_datos_entrenamiento}
            prediccion_1 = prediccion_0 * ({float(factor_04 == 0)}d0) + fis_tsukamoto(base, params, prediccion_0, x2(i))

            suma_cuadrados = suma_cuadrados + (respuesta(i) - prediccion_1)**2
            SSE = SSE + (respuesta(i) - prediccion_1)**2
            RMSE = sqrt(SSE / i)
            if (RMSE > RMSE_crit) then
                RMSE = sqrt(SSE + ({float(numero_datos_entrenamiento)}d0 - i) * RMSE_crit ** 2) / sqrt({float(numero_datos_entrenamiento)}d0)
                exit
            end if
            prediccion_0 = prediccion_1
        end do
        objetivo = RMSE

    end function entrenamiento
'''

mamdani_entrenamiento = f'''
    function entrenamiento(x) result(objetivo)
        double precision, dimension(:), intent(in) :: x
        double precision, dimension({numero_datos_entrenamiento}) :: x1, x2, respuesta
        double precision, dimension(3,3,3) :: base
        double precision, dimension(34) :: params
        double precision :: suma_cuadrados, prediccion_0, prediccion_1, SSE
        double precision :: RMSE, objetivo, RMSE_crit
        double precision, dimension(3) :: antecedente01, antecedente02
        integer :: i

    {importar_datos}
        
        suma_cuadrados = 0.0d0 
        RMSE_crit = {f_objetivo_critico}d0
        SSE = 0.0d0
        {call_rules}

{definicion_parametros_mamdani}

        prediccion_0 = x1(1)
        do i = 1, {numero_datos_entrenamiento}
            prediccion_1 = prediccion_0 * ({float(factor_04 == 0)}d0) + fis_mamdani(base, params, prediccion_0, x2(i))

            suma_cuadrados = suma_cuadrados + (respuesta(i) - prediccion_1)**2
            SSE = SSE + (respuesta(i) - prediccion_1)**2
            RMSE = sqrt(SSE / i)
            if (RMSE > RMSE_crit) then
                RMSE = sqrt(SSE + ({float(numero_datos_entrenamiento)}d0 - i) * RMSE_crit ** 2) / sqrt({float(numero_datos_entrenamiento)}d0)
                exit
            end if
            prediccion_0 = prediccion_1
        end do
        ! print *, "RMSE = ", RMSE
        objetivo = RMSE

    end function entrenamiento
'''

utiles = f'''
    double precision function trapezoidal(params, w)
        double precision, dimension(4), intent(in) :: params
        double precision, intent(in) :: w
        double precision :: a, b, c, d
        a = params(1)
        b = params(2)
        c = params(3)
        d = params(4)
        trapezoidal = max(min((w - a) / (b - a), min(1.0, (d - w) / (d - c))), 0.0)
    end function trapezoidal 

    double precision function triangular(params, w)
        double precision, dimension(3), intent(in) :: params
        double precision, intent(in) :: w 
        double precision :: a, b, c
        a = params(1)
        b = params(2)
        c = params(3)
        triangular = max(min((w - a) / (b - a), (c - w) / (c - b)), 0.0)
    end function triangular 

    double precision function lineal(x0, x1, y)
        double precision, intent(in) :: x0, x1, y
        lineal = y * (x1 - x0) + x0
    end function lineal

    subroutine sort_real_asc(a)
        double precision, intent(inout) :: a(:)
        integer :: i, j
        double precision :: clave
        do i = 2, size(a)
            clave = a(i)
            j = i - 1
            do while (j >= 1 .and. a(j) > clave)
                a(j+1) = a(j)
                j = j - 1
            end do
            a(j+1) = clave
        end do
    end subroutine
'''

differential_evolution = f'''
    function differential_evolution() result(Solution)
        double precision, dimension(19) :: Solution
        double precision, dimension(17) :: LimInf, LimSup
        double precision :: F, Cr, Fbest, FunZ, rand_num
        integer :: i, j, k, n, r1, r2, r3, ultima_actualizacion, calls
        double precision, dimension({tamanio_poblacion}, 17) :: X
        double precision, dimension({tamanio_poblacion}) :: Fit
        double precision, dimension(17) :: Xbest, z

        LimInf = 0.0d0
        LimSup = 50.0d0
        LimInf(1:9) = 1.0d0
        LimSup(1:9) = 3.0d0

        F = 0.5d0
        Cr = 0.2d0
        Fbest = 1.0d24
        Xbest = 0.0d0

        calls = 0
        ultima_actualizacion = {tamanio_poblacion * numero_generaciones}

        do i = 1, {tamanio_poblacion}
            do j = 1, 17
                call random_number(rand_num)
                X(i, j) = LimInf(j) + (LimSup(j) - LimInf(j)) * rand_num
            end do
            Fit(i) = entrenamiento(X(i, :))
            calls = calls + 1
        end do

        do i = 1, {numero_generaciones}
            do j = 1, {tamanio_poblacion}
                call random_number(rand_num)
                r1 = int(rand_num * {tamanio_poblacion}) + 1
                do 
                    call random_number(rand_num)
                    r2 = int(rand_num * {tamanio_poblacion}) + 1
                    if (r2 /= r1) exit 
                end do 
                do 
                    call random_number(rand_num)
                    r3 = int(rand_num * {tamanio_poblacion}) + 1
                    if (r3 /= r1 .and. r3 /= r2) exit 
                end do 
                z = X(j, :)
                do k = 1, 17
                    call random_number(rand_num)
                    if (rand_num < Cr) then 
                        z(k) = X(r1,k) + F * (X(r2,k)-X(r3,k))
                    end if
                    if (z(k) > LimSup(k) .or. z(k) < LimInf(k)) then 
                        do n = 1, 17
                            call random_number(rand_num)
                            z(n) = LimInf(n) + (LimSup(n) - LimInf(n)) * rand_num
                        end do
                        exit 
                    end if 
                end do
                FunZ = entrenamiento(z)
                calls = calls + 1
                if (FunZ < Fit(j)) then 
                    X(j, :) = z 
                    Fit(j) = FunZ 
                    if (FunZ < Fbest) then 
                        Fbest = FunZ
                        Xbest = z 
                    end if 
                end if
            end do
            if (Fbest < 0.0d0) then 
                ultima_actualizacion = calls
                exit 
            end if 
        end do
        Solution(1:17) = Xbest
        Solution(18) = Fbest
        Solution(19) = dble(ultima_actualizacion)
    end function differential_evolution
'''

if factor_02 == 0:
    modulo_optimizacion = f'''
module optimizacion
    implicit none
contains
{mamdani_entrenamiento}

{funcion_base_de_reglas}

{funcion_fis_mamdani}

{funcion_x_bar_trapecio}

{funcion_area_trapecio}

{funcion_zona_conficto}

{funcion_centroide_total}

{utiles}

{differential_evolution}
end module optimizacion
'''
else: 
    modulo_optimizacion = f'''
module optimizacion
    implicit none
contains
{funcion_base_de_reglas}

{funcion_fis_tsukamoto}

{tsukamoto_entrenamiento}

{utiles}

{differential_evolution}
end module optimizacion
'''


optimizacion_driver = f'''
program OptimizationDriver
! gfortran -c optimizacion.f90
! gfortran -c OptimizationDriver.f90
! gfortran optimizacion.o OptimizationDriver.o -o driver.exe

    use optimizacion
    implicit none

    integer :: iteracion
    double precision, dimension({pruebas},19) :: soluciones
    double precision, dimension(19) :: solucion
    character(len=100) :: nombre_archivo

    nombre_archivo = "experimento_{factor_01}_{factor_02}_{factor_03}_{factor_04}.csv"
    solucion = 1.0d0        
    do iteracion = 1, {pruebas}
        solucion = differential_evolution()
        soluciones(iteracion, :) = solucion
        print *, "Resultado de la iteración ", iteracion, ": ", solucion
    end do

    ! Guarda los resultados en un archivo CSV
    open(unit=10, file=trim(nombre_archivo), status="replace")
    ! Escribe encabezados en el archivo CSV

    write(10, '(A)') "x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,Costo,Times"

        do iteracion = 1, {pruebas}
            write(10, '(F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7,",",F15.7)') &
            soluciones(iteracion, 1), soluciones(iteracion, 2), soluciones(iteracion, 3), soluciones(iteracion, 4), &
            soluciones(iteracion, 5), soluciones(iteracion, 6), soluciones(iteracion, 7), soluciones(iteracion, 8), &
            soluciones(iteracion, 9), soluciones(iteracion,10), soluciones(iteracion,11), soluciones(iteracion,12), &
            soluciones(iteracion,13), soluciones(iteracion,14), soluciones(iteracion,15), soluciones(iteracion,16), &
            soluciones(iteracion,17), soluciones(iteracion,18), soluciones(iteracion,19)
        end do
    close(10)
    print *, "Resultados guardados en: ", trim(nombre_archivo)
end program OptimizationDriver
'''

documento_sh = f'''
gfortran -c optimizacion.f90
gfortran -c OptimizationDriver.f90
gfortran optimizacion.o OptimizationDriver.o -o driver_{factor_01}{factor_02}{factor_03}{factor_04}.exe
'''

with open("optimizacion.f90", "w", encoding="utf-8") as f:
    f.write(modulo_optimizacion)

with open("OptimizationDriver.f90", "w", encoding="utf-8") as f:
    f.write(optimizacion_driver)

with open("runner.sh", "w", encoding="utf-8") as f:
    f.write(documento_sh)
