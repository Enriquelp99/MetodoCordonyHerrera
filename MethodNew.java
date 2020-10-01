/** *********************************************************************
 *
 * This file is part of KEEL-software, the Data Mining tool for regression,
 * classification, clustering, pattern mining and so on.
 *
 * Copyright (C) 2004-2010
 *
 * F. Herrera (herrera@decsai.ugr.es)
 * L. Sánchez (luciano@uniovi.es)
 * J. Alcalá-Fdez (jalcala@decsai.ugr.es)
 * S. García (sglopez@ujaen.es)
 * A. Fernández (alberto.fernandez@ujaen.es)
 * J. Luengo (julianlm@decsai.ugr.es)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/
 *
 ********************************************************************* */
package keel.Algorithms.RE_SL_Methods.MethodNew;

import static java.lang.Math.pow;
import java.util.*;

class MethodNew {

    public int MaxReglas = 10000;

    public double[] grado_pertenencia;
    public int[] Regla_act;
    public Vector<TipoRegla> Conjunto_Reglas;
    public int peso;

    public String fich_datos_chequeo, fich_datos_tst;
    public String fichero_conf, ruta_salida;
    public String fichero_reglas, fichero_inf, fich_tra_obli, fich_tst_obli;
    public String informe = "";
    public String cadenaReglas = "";

    public MiDataset tabla, tabla_tst;
    public BaseR base_reglas;
    public BaseD base_datos;
    public Adap fun_adap;

    public int tipoFVR;
    public int netiqueta;
    public ArrayList<ArrayList> subespacios;
    public ArrayList<ArrayList> grado_pertenecia_ejemplos;
    public Vector<Double> FVRsubespacios;

    public MethodNew(String f_e) {
        fichero_conf = f_e;

    }

    private String Quita_blancos(String cadena) {
        StringTokenizer sT = new StringTokenizer(cadena, "\t ", false);
        return (sT.nextToken());
    }

    /**
     * Reads the data of the configuration file
     */
    public void leer_conf() {
        int i, n_etiquetas;
        String cadenaEntrada, valor;

        // we read the file in a String
        cadenaEntrada = Fichero.leeFichero(fichero_conf);
        StringTokenizer sT = new StringTokenizer(cadenaEntrada, "\n\r=", false);

        // we read the algorithm's name
        sT.nextToken();
        sT.nextToken();

        // we read the name of the training and test files
        sT.nextToken();
        valor = sT.nextToken();

        StringTokenizer ficheros = new StringTokenizer(valor, "\t ", false);
        ficheros.nextToken();
        fich_datos_chequeo = ((ficheros.nextToken()).replace('\"', ' ')).trim();
        fich_datos_tst = ((ficheros.nextToken()).replace('\"', ' ')).trim();

        // we read the name of the output files
        sT.nextToken();
        valor = sT.nextToken();

        ficheros = new StringTokenizer(valor, "\t ", false);
        fich_tra_obli = ((ficheros.nextToken()).replace('\"', ' ')).trim();
        fich_tst_obli = ((ficheros.nextToken()).replace('\"', ' ')).trim();
        fichero_reglas = ((ficheros.nextToken()).replace('\"', ' ')).trim();
        fichero_inf = ((ficheros.nextToken()).replace('\"', ' ')).trim();
        ruta_salida = fichero_reglas.substring(0,
                fichero_reglas.lastIndexOf('/')
                + 1);

        // we read the Number of labels
        sT.nextToken();
        valor = sT.nextToken();
        n_etiquetas = Integer.parseInt(valor.trim());
        netiqueta = n_etiquetas;

        // we read the KB Output File Format with Weight values to 1 (0/1)
        peso = 0;

        // we create all the objects
        tabla = new MiDataset(fich_datos_chequeo, true);
        tabla_tst = new MiDataset(fich_datos_tst, false);
        base_datos = new BaseD(n_etiquetas, tabla.n_variables);

        for (i = 0; i < tabla.n_variables; i++) {
            base_datos.n_etiquetas[i] = n_etiquetas;
            base_datos.extremos[i].min = tabla.extremos[i].min;
            base_datos.extremos[i].max = tabla.extremos[i].max;
        }

        MaxReglas = tabla.long_tabla; //MaxReglas == #Ejemplos
        base_reglas = new BaseR(MaxReglas, base_datos, tabla);
        fun_adap = new Adap(tabla, tabla_tst, base_reglas);
        Regla_act = new int[tabla.n_variables];
        grado_pertenencia = new double[tabla.n_variables];
        Conjunto_Reglas = new Vector<TipoRegla>();

        for (i = 0; i < tabla.long_tabla; i++) {
            Conjunto_Reglas.add(i, new TipoRegla(tabla.n_variables));
        }
    }

    public void run() {
        int i, j, k, etiqueta, pos;
        double pert_act, grado_act, ec, el, ec_tst, el_tst;

        /* We read the configutate file and we initialize the structures and variables */
        leer_conf();

        if (tabla.salir == false) {
            /* we generate the semantics of the linguistic variables */
            base_datos.Semantica();

            /* we store the DB in the report file */
            informe = "\n\nInitial Data Base: \n\n";
            for (i = 0; i < tabla.n_variables; i++) {
                informe += "  Variable " + (i + 1) + ":\n";
                for (j = 0; j < base_datos.n_etiquetas[i]; j++) {
                    informe += "    Label " + (j + 1) + ": ("
                            + base_datos.BaseDatos[i][j].x0 + ","
                            + base_datos.BaseDatos[i][j].x1 + ","
                            + base_datos.BaseDatos[i][j].x3 + ")\n";
                }

                informe += "\n";
            }

            informe += "\n";
            Fichero.escribeFichero(fichero_inf, informe);

            /* Inicialization of the counter of uncovered examples */
            base_reglas.n_reglas = 0;

            double tamsubespacios = pow(netiqueta, tabla.n_variables);

            subespacios = new ArrayList<ArrayList>();

            int tamsubespacios2 = (int) tamsubespacios;
            int contador = 0;

            for (int inicio = 0; inicio < tamsubespacios; inicio++) {
                FVRsubespacios = new Vector<Double>(netiqueta);
                ArrayList<Integer> indices = obtenerVariablesyEtiquetas(netiqueta, tabla.n_variables, inicio);
                ArrayList<Double> EVejemplos = new ArrayList<Double>();
                boolean antecesores = true;
                if (!indices.isEmpty()) {
                    //CONJUNTO DE EJEMPLOS
                    for (i = 0; i < tabla.long_tabla; i++) {
                        double VE = Double.MAX_VALUE;
                        //VARIABLES
                        for (int indice = tabla.n_variables - 1; indice > 0; indice--) {
                            if (antecesores) {
                                int _variable = indice;
                                int _etiqueta = indices.get(indice);
                                pert_act = base_reglas.Fuzzifica(tabla.datos[i].ejemplo[_variable], base_datos.BaseDatos[_variable][_etiqueta]);
                                if (pert_act <= 0) {
                                    antecesores = false;
                                    VE = Double.MAX_VALUE;
                                } else {
                                    if (pert_act < VE) {
                                        VE = pert_act;
                                    }
                                }
                                EVejemplos.add(VE);
                            } else {
                                double cero = 0;
                                EVejemplos.add(cero);
                            }
                        }
                    }
                    tipoFVR = 1;
                    if (tipoFVR == 0) {
                        double mayor = 0;
                        int posmayor = 0;

                        for (int totalFVR = 0; totalFVR < EVejemplos.size() - 1; totalFVR++) {
                            if (mayor < EVejemplos.get(totalFVR)) {
                                mayor = EVejemplos.get(totalFVR);
                                posmayor = totalFVR;
                            }
                        }
                        FVRsubespacios.add(mayor);

                    } else {
                        if (tipoFVR == 1) {
                            double sumatorio = 0;
                            double total = 0;
                            for (int totalFVR = 0; totalFVR < EVejemplos.size(); totalFVR++) {
                                sumatorio += EVejemplos.get(totalFVR);
                            }
                            total = (sumatorio / EVejemplos.size());

                            FVRsubespacios.add(total);
                        }
                    }
                }

                if (contador < netiqueta) {
                    if (contador == netiqueta - 1) {

                        ArrayList<Integer> indices3 = obtenerVariablesyEtiquetas(netiqueta, tabla.n_variables, inicio - contador);
                        int posicion = inicio - contador;
                        double mayor = 0;
                        int posmayor = 0;
                        if (!indices.isEmpty()) {
                            for (int iFVR = 0; iFVR < FVRsubespacios.size(); iFVR++) {
                                double valor = FVRsubespacios.get(iFVR);
                                if (valor > mayor) {
                                    mayor = valor;
                                    posmayor = posicion + iFVR;
                                }
                            }
                            ArrayList<Integer> indices2 = obtenerVariablesyEtiquetas(netiqueta, tabla.n_variables, posmayor);
                            int[] Regla = new int[tabla.n_variables];

                            int valores = 0;

                            for (int indiceFVR = indices2.size() - 1; indiceFVR >= 0; indiceFVR--) {
                                Regla[valores] = indices2.get(indiceFVR);

                                valores++;
                            }
                            if (base_reglas.n_reglas <tabla.long_tabla) {
                                TipoRegla conjunto = Conjunto_Reglas.get(base_reglas.n_reglas);
                                for (j = 0; j < tabla.n_variables; j++) {
                                    conjunto.Regla.add(Regla[j]);
                                }
                                conjunto.grado = mayor;
                                Conjunto_Reglas.remove(base_reglas.n_reglas);
                                Conjunto_Reglas.add(base_reglas.n_reglas, conjunto);
                                base_reglas.n_reglas++;
                            }
                        }
                        contador = 0;
                    } else {
                        contador++;
                    }
                }

            }

            /* we decode the generated rules *
        
            /* we calcule the MSEs */
            base_reglas.decodifica(Conjunto_Reglas);
            fun_adap.Error_tra();
            ec = fun_adap.EC;
            el = fun_adap.EL;

            fun_adap.Error_tst();
            ec_tst = fun_adap.EC;
            el_tst = fun_adap.EL;

            /* we write the RB */
            cadenaReglas = base_reglas.BRtoString(peso);
//            cadenaReglas += "\n";
//            cadenaReglas += "\nECMtra: " + ec + " ECMtst: " + ec_tst + "\n";

            Fichero.escribeFichero(fichero_reglas, cadenaReglas);

            /* we write the obligatory output files*/
            String salida_tra = tabla.getCabecera();
            salida_tra += fun_adap.getSalidaObli(tabla);
            Fichero.escribeFichero(fich_tra_obli, salida_tra);

            String salida_tst = tabla_tst.getCabecera();
            salida_tst += fun_adap.getSalidaObli(tabla_tst);
            Fichero.escribeFichero(fich_tst_obli, salida_tst);

            /* we write the MSEs in specific files */
            Fichero.AnadirtoFichero(ruta_salida + "MethodNewcomunR.txt",
                    "" + base_reglas.n_reglas + "\n");
            Fichero.AnadirtoFichero(ruta_salida + "MethodNewcomunTRA.txt",
                    "" + 2.0 * ec + "\n");
            Fichero.AnadirtoFichero(ruta_salida + "MethodNewcomunTST.txt",
                    "" + 2.0 * ec_tst + "\n");
        }
    }

    /**
     * Returns 1 if the better current rule is in the list "L" yet
     *
     * @param R int[] New rule to check
     * @param L TipoRegla[] List of rules
     * @param n_generadas int Total number of rules
     * @return int 1 if the better current rule is in the list "L" yet, -1 else.
     */
    int Pertenece(int[] R, TipoRegla[] L, int n_generadas) {
        int nreg, var, esta;

        nreg = 0;
        while (nreg < n_generadas) {
            esta = 1;
            var = 0;

            while (var < tabla.n_var_estado && esta == 1) {
                if (R[var] != L[nreg].Regla.get(var)) {
                    esta = 0;
                } else {
                    var++;
                }
            }

            if (esta == 1) {
                return (nreg);
            }
            nreg++;
        }

        return (-1);
    }

    private ArrayList<Integer> obtenerVariablesyEtiquetas(int n_etiqueta, int n_variables, int subespacio) {
        int n = subespacio;
        int valor, disminuir;
        ArrayList<Integer> index = new ArrayList<Integer>(n_variables);
        for (int i = 0; i < n_variables; i++) {
            valor = n % 10;
            if (valor < n_etiqueta) {
                index.add(i, valor);
                n = n / 10;
            } else {
                return new ArrayList<Integer>();
            }

        }
        return index;
    }
}
