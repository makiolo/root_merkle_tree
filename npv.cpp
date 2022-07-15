// https://www.linkedin.com/pulse/python-bootstrapping-zero-curve-sheikh-pancham#:~:text=The%20objective%20of%20bootstrapping%20is,the%20end%20of%20its%20term.

#include <iostream>
#include <cmath>
#include <vector>
#include <catch_amalgamated.hpp>

enum Convention
{
    LINEAR,
    YIELD,
    EXPONENTIAL,
};

double fv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD);

double round3(double var)
{
    char str[10];
    sprintf(str, "%.3f", var);
    var = strtof(str, nullptr);
    return var;
}

double df2zc(double df, double maturity, int compound_times = 1, Convention conv = Convention::YIELD)
{
    switch (conv)
    {
    case Convention::LINEAR:
        return (1.0 / df - 1.0) * (1.0 / maturity);
    case Convention::YIELD:
        return (pow(1.0 / df, 1.0 / (maturity * compound_times)) - 1.0) * compound_times;
    case Convention::EXPONENTIAL:
        return -log(df) / maturity;
    default:
        throw std::runtime_error("Invalid convention");
    }
}

double zc2df(double zc, double maturity, int compound_times = 1, Convention conv = Convention::YIELD)
{
    switch (conv)
    {
    case Convention::LINEAR:
        return 1.0 / (1.0 + zc * maturity);
    case Convention::YIELD:
        return 1.0 / (pow((1.0 + zc / compound_times), maturity * compound_times));
    case Convention::EXPONENTIAL:
        return exp(-zc * maturity);
    default:
        throw std::runtime_error("Invalid convention");
    }
}

double equivalent_rate(double rate, int compound_times, Convention convention = Convention::YIELD, int other_compound_times = 1, Convention other_convention = Convention::YIELD)
{
    double df = zc2df(rate, 1, compound_times, convention);
    return df2zc(df, 1, other_compound_times, other_convention);
}

double equivalent_rate(double rate, int compound_times, int other_compound_times = 1)
{
    //return other_compound_times * pow(1.0 + (rate / compound_times), double(compound_times) / other_compound_times) - other_compound_times;
    return equivalent_rate(rate, compound_times, Convention::YIELD, other_compound_times, Convention::YIELD);
}

std::vector<double> get_discount_factors_1_T(double r, double years, int compound_times = 1, Convention convention = Convention::YIELD)
{
    std::vector<double> dfs;
    for (int i = 1; i <= years * compound_times; ++i)
    {
        // Z(x) = Z(i / compound_times)
        double df = zc2df(r, double(i) / compound_times, compound_times, convention);
        // df = round3(df);
        dfs.push_back(df);
    }
    return dfs;
}

std::vector<double> get_discount_factors_0_T_less1(double r, double years, int compound_times = 1, Convention convention = Convention::YIELD)
{
    std::vector<double> dfs;
    for (int i = 0; i < years * compound_times; ++i)
    {
        // Z(x) = Z(i / compound_times)
        double df = zc2df(r, double(i) / compound_times, compound_times, convention);
        dfs.push_back(df);
    }
    return dfs;
}

// tenemos un cash en "maturity" y nos lo traemos a presente
double to_present_value(double cash, double r, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double df = zc2df(r, maturity, compound_times, convention);
    return cash * df;
}

// tenemos un cash en "maturity" y nos lo traemos a futuro 
double to_future_value(double cash, double r, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double df = zc2df(r, maturity, compound_times, convention);
    return cash / df;
}

double real_from_coupon(double coupon, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    return coupon * maturity * compound_times;
}

double coupon_from_real(double real, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    return real / (maturity * compound_times);
}

// only coupons
double npv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double npv = 0.0;
    for (const auto& df : dfs)
    {
        npv += df * (coupon / compound_times);
    }
    return npv;
}

double npv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD, int g_compound_times = 1, Convention g_convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double npv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        npv += (df * (coupon / compound_times)) / zc2df(g, i, g_compound_times, g_convention);
        i += 1.0;
    }
    return npv;
}

double coupon_from_npv(double npv, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += df;
    }
    return (npv * compound_times) / total_df;
}

// VAN = coupons - initial investement
double classic_npv(double investment, double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupons = npv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    double npv = coupons - investment;
    return npv;
}

// coupons + payment on yield-maturity
double bond_npv(double face_value, double coupon, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupons = npv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    double yield_on_payment = to_present_value(face_value, interest_rate, maturity, compound_times, convention);
    return coupons + yield_on_payment;
}

// stock investment = coupons + payment on yield-maturity - initial investement
double stock_npv(double investment, double dividend, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupon = investment * (dividend - interest_rate);
    double coupons = npv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    double yield_on_payment = to_present_value(investment, interest_rate, maturity, compound_times, convention);
    double npv = coupons + yield_on_payment - investment;
    return npv;
}

double stock_fv(double investment, double dividend, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupon = investment * (dividend - interest_rate);
    //double investment_fv = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    double coupons = fv_from_coupon(coupon, interest_rate, maturity, compound_times, convention);
    //double yield_on_payment = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    //double npv = coupons + yield_on_payment - investment_fv;
    double npv = coupons;
    return npv;
}

double stock_real(double investment, double dividend, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    double coupon = investment * (dividend - interest_rate);
    double investment_fv = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    double coupons = real_from_coupon(coupon, maturity, compound_times, convention);
    //double yield_on_payment = to_future_value(investment, interest_rate, maturity, compound_times, convention);
    //double npv = coupons + yield_on_payment - investment_fv;
    double npv = investment - coupons - investment_fv;
    return npv;
}

double fv_from_coupon(double coupon, double interest_rate, double maturity, int compound_times, Convention convention)
{
    auto dfs = get_discount_factors_0_T_less1(interest_rate, maturity, compound_times, convention);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += (1.0 / df);
    }
    return coupon * total_df;
}

/*
double fv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD, int g_compound_times = 1, Convention g_convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_1_T(interest_rate, maturity, compound_times, convention);
    double npv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        npv += ((1.0 / df) * (coupon / compound_times)) / zc2df(g, i, g_compound_times, g_convention);
        i += 1.0;
    }
    return npv;
}
*/

double coupon_from_fv(double fv, double interest_rate, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    auto dfs = get_discount_factors_0_T_less1(interest_rate, maturity, compound_times, convention);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += 1.0 / df;
    }
    double coupon = fv / total_df;
    return coupon;
}

double cagr(double initial, double final, double maturity = 1.0)
{
    return pow(final / initial, 1.0 / maturity) - 1.0;
}

double interest_on_capital(double initial, double final, double maturity, int compound_times = 1, Convention convention = Convention::YIELD)
{
    if (compound_times == 1 and convention == Convention::YIELD)
    {
        return cagr(initial, final, maturity);
    }
    else
    {
        double r = (final - initial) / initial;
        double df = zc2df(r, 1, 1, Convention::LINEAR);  // 1.0 / (1.0 + zc);
        return df2zc(df, maturity, compound_times, convention);  // -log(df) / maturity   etc ...
    }
}

TEST_CASE("bond_npv", "[npv]") {

    // Comparado con: https://mathcracker.com/es/calculadora-valor-bonos#results
    // valor presente de un bono
    // valorar un bono que da un yield "seguro" haciendo otros proyectos risk free
    double npv = bond_npv(16000,
        // inversion "segura" ofrecida por el bono
        100, 0.06,
        // inversion libre de riesgo en el mercado
        20, 1, Convention::YIELD);

    REQUIRE(npv == Catch::Approx(6135.87));
}

TEST_CASE("fv_from_coupon", "[npv]") {

    // Ahorro inicial en el futuro

    double initial = 10000;
    double r = 0.07;
    double maturity = 8;
    double fv1 = to_future_value(initial, r, maturity, 1, Convention::YIELD);
    double aportado1 = initial;
    double presente1 = to_present_value(initial, r, 0, 1, Convention::YIELD);

    REQUIRE(aportado1 == Catch::Approx(10000));
    REQUIRE(presente1 == Catch::Approx(10000));
    REQUIRE(fv1 == Catch::Approx(17181.8617983192));

    // Ahorro periodico (anual)

    double cuota = 5000;
    double fv2 = fv_from_coupon(5000, r, maturity, 1, Convention::YIELD);
    double aportado2 = real_from_coupon(5000, maturity, 1, Convention::YIELD);
    double presente2 = npv_from_coupon(5000, r, maturity, 1, Convention::YIELD);

    REQUIRE(aportado2 == Catch::Approx(40000.0));
    REQUIRE(presente2 == Catch::Approx(29856.4925310687));
    REQUIRE(fv2 == Catch::Approx(51299.0128451372));

    // Ahorro periodico (mensual)

    cuota = 1000;
    double compound_times = 12;
    double fv3 = fv_from_coupon(1000, r, maturity, compound_times, Convention::YIELD);
    double aportado3 = real_from_coupon(1000, maturity, compound_times, Convention::YIELD);
    double presente3 = npv_from_coupon(1000, r, maturity, compound_times, Convention::YIELD);

    REQUIRE(presente3 == Catch::Approx(6112.297390453));
    REQUIRE(aportado3 == Catch::Approx(96000.0));
    REQUIRE(fv3 == Catch::Approx(128198.8210340072));

    double final;
    double presente_total;
    initial = aportado1 + aportado2 + aportado3;
    presente_total = presente1 + presente2 + presente3;
    final = fv1 + fv2 + fv3;
    REQUIRE(presente_total == Catch::Approx(45968.7899215216));
    REQUIRE(coupon_from_real(initial, maturity, 12, Convention::YIELD) == Catch::Approx(1520.8333333333));
    REQUIRE(coupon_from_real(initial, maturity, 1, Convention::YIELD) == Catch::Approx(18250.0));
    REQUIRE(initial == Catch::Approx(146000.0));
    REQUIRE(final == Catch::Approx(196679.6956774635));

    double r_invest = cagr(initial, final, maturity);
    REQUIRE(r_invest == Catch::Approx(0.0379485678));
}

TEST_CASE("cagr_interest", "[npv]") {

    double initial = 11000;
    double final = 21000;
    double past_years = 2;
    double forward_years = 2;
    // past info
    double r = interest_on_capital(initial, final, past_years, 1, Convention::YIELD);
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    r = cagr(initial, final, past_years);
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    // forward prediction
    REQUIRE(to_future_value(final, r, forward_years, 12, Convention::YIELD) == Catch::Approx(44524.0670913586));

    // trading
    initial = 5000;
    r = 0.10;
    double r_anual = equivalent_rate(r, 12, 1);
    double years = 3.0;
    REQUIRE(to_future_value(initial, r, years, 12, Convention::YIELD) == \
        Catch::Approx(to_future_value(initial, r_anual, years, 1, Convention::YIELD)));
}

TEST_CASE("df & zc", "[npv]") {

    REQUIRE(zc2df(df2zc(0.95, 3, 1, Convention::LINEAR), 3, 1, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 1, Convention::LINEAR), 3, 1, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 4, Convention::LINEAR), 3, 4, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 4, Convention::LINEAR), 3, 4, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 1, Convention::YIELD), 3, 1, Convention::YIELD) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 1, Convention::YIELD), 3, 1, Convention::YIELD) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 4, Convention::YIELD), 3, 4, Convention::YIELD) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 4, Convention::YIELD), 3, 4, Convention::YIELD) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 1, Convention::EXPONENTIAL), 3, 1, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 1, Convention::EXPONENTIAL), 3, 1, Convention::EXPONENTIAL) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, 4, Convention::EXPONENTIAL), 3, 4, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, 4, Convention::EXPONENTIAL), 3, 4, Convention::EXPONENTIAL) == Catch::Approx(0.05));
}

TEST_CASE("bond_npv2", "[npv]") {

    double cash = 17181.8617983192;
    double r = 0.07;
    double maturity = 8;
    REQUIRE(to_present_value(cash, r, maturity, 1, Convention::YIELD) == Catch::Approx(10000));

    // future value
    double fv = 51299.0128451372;
    REQUIRE(coupon_from_fv(fv, r, maturity, 1, Convention::YIELD) == Catch::Approx(5000));
    REQUIRE(fv_from_coupon(5000, r, maturity, 1, Convention::YIELD) == Catch::Approx(fv));

    // traer flujos futuros a presente
    double npv = npv_from_coupon(5000, r, maturity, 1, Convention::YIELD);
    REQUIRE(npv == Catch::Approx(29856.4925310687));

    // Traerme a presente flujos futuros anuales
    REQUIRE(coupon_from_npv(npv, r, maturity, 1, Convention::YIELD) == Catch::Approx(5000));

    // implementar VAN y TIR
    REQUIRE(classic_npv(6000,
        // cuota y free risk rate
        200, 0.08,
        // maturity y tipo de porcentaje
        3, 1, Convention::YIELD) == Catch::Approx(-5484.5806025504));
}

TEST_CASE("real coupon", "[npv]") {

    double coupon_netflix = 9.9;
    double maturity = 10;
    double real = real_from_coupon(coupon_netflix, maturity, 12, Convention::YIELD);

    REQUIRE(real == Catch::Approx(1188.0));

    // dividendo 0.08, precio dinero 0.03

    double npv = stock_npv(1000, 0.08, 0.03, maturity, 1, Convention::YIELD);
    REQUIRE(npv == Catch::Approx(170.6040567355));

    double fv = stock_fv(1000, 0.08, 0.03, maturity, 1, Convention::YIELD);
    REQUIRE(fv == Catch::Approx(573.1939655735));

    double real2 = stock_real(1000, 0.08, 0.03, maturity, 1, Convention::YIELD);
    REQUIRE(real2 == Catch::Approx(-843.9163793441));

    // dividendo 0.08, precio dinero 0.12

    double npv_ = stock_npv(1000, 0.08, 0.12, maturity, 1, Convention::YIELD);
    REQUIRE(npv_ == Catch::Approx(-904.0356845457));

    double fv_ = stock_fv(1000, 0.08, 0.12, maturity, 1, Convention::YIELD);
    REQUIRE(fv_ == Catch::Approx(-701.9494027814));

    double real2_ = stock_real(1000, 0.08, 0.12, maturity, 1, Convention::YIELD);
    REQUIRE(real2_ == Catch::Approx(-1705.8482083442));

    REQUIRE(interest_on_capital(npv, fv, maturity, 1, Convention::EXPONENTIAL) == Catch::Approx(0.1211878754));
    REQUIRE(interest_on_capital(npv_, fv_, maturity, 1, Convention::EXPONENTIAL) == Catch::Approx(-0.0253007508));
}

TEST_CASE("tn & te", "[npv]")
{
    double a = 0.05 / 12;
    // TASA NOMINAL a TASA EFECTIVA
    double b = equivalent_rate(0.05, 12, 1) / 12;
    // TASA EFECTIVA A TASA NOMINAL
    double c = equivalent_rate(0.05, 1, 12) / 12;

    double c1 = 1000 * a;
    double c2 = 1000 * b;
    double c3 = 1000 * c;

    REQUIRE(c1 == Catch::Approx(4.1666666667));
    REQUIRE(c2 == Catch::Approx(4.2634914901));
    REQUIRE(c3 == Catch::Approx(4.0741237836));

    // 5% reinvirtiendo 1 vez al añao
    REQUIRE(cagr(1000, 1000 + (c1 * 12)) == Catch::Approx(0.05));
    // 5% reinvirtiendo 12 veces equivalen a 5.1161% reinvirtiendo 1
    REQUIRE(cagr(1000, 1000 + (c2 * 12)) == Catch::Approx(0.0511618979));
    // 5% reinvirtiendo 1 vez equivalen a 4.888% reinvirtiendo 12
    REQUIRE(cagr(1000, 1000 + (c3 * 12)) == Catch::Approx(0.0488894854));

    //REQUIRE(tn_2_te(0.05, 12) == Catch::Approx(0.0511618979));
    REQUIRE(equivalent_rate(0.05, 12, 1) == Catch::Approx(0.0511618979));

    //REQUIRE(te_2_tn(0.05, 12) == Catch::Approx(0.0488894854));
    REQUIRE(equivalent_rate(0.05, 1, 12) == Catch::Approx(0.0488894854));
    
    REQUIRE(equivalent_rate(0.0488894854, 12, 1) == Catch::Approx(0.05));
    REQUIRE(equivalent_rate(0.0511618979, 1, 12) == Catch::Approx(0.05));

    REQUIRE(equivalent_rate(0.01, 365, 1) == Catch::Approx(0.0100500287));
    REQUIRE(equivalent_rate(0.01, 1, 365) == Catch::Approx(0.0099504665));

    /*
    10% mensual con reinversion mensual
    */
    double  fv = to_future_value(1000, 0.10 * 12, 1, 12, Convention::YIELD);
    REQUIRE(fv == Catch::Approx(3138.428376721));
    REQUIRE(cagr(1000, fv) == Catch::Approx(equivalent_rate(0.10 * 12, 12, 1)));

    /*
    10% mensual con reinversion anual = 120%
    */
    double  fv2 = to_future_value(1000, 0.10 * 12, 1, 1, Convention::YIELD);
    REQUIRE(fv2 == Catch::Approx(2200.0));
    REQUIRE(cagr(1000, fv2) == Catch::Approx(equivalent_rate(0.10 * 12, 1, 1)));

    /*
    2% semanal con reinversion semanal = 191.34%
    */
    double  fv3 = to_future_value(1000, 0.02 * 54, 1, 54, Convention::YIELD);
    REQUIRE(fv3 == Catch::Approx(2913.4614441403));
    REQUIRE(cagr(1000, fv3) == Catch::Approx(equivalent_rate(0.02 * 54, 54, 1)));

    /*
    2% semanal con reinversion continua = 194.46%
    */
    double  fv4 = to_future_value(1000, 0.02 * 54, 1, 1, Convention::EXPONENTIAL);
    REQUIRE(fv4 == Catch::Approx(2944.6795510655));
    // ¿Como calcular ese CAGR?
    REQUIRE(cagr(1000, fv4) == Catch::Approx(equivalent_rate(0.02 * 54, 1, Convention::EXPONENTIAL, 1, Convention::YIELD)));

    REQUIRE(equivalent_rate(0.05, 1, 12) == equivalent_rate(0.05, 1, Convention::YIELD, 12, Convention::YIELD));
}

TEST_CASE("coupon growth", "[npv]")
{
    // el dividendo no crece
    double npv1 = npv_from_coupon(1000, 0.08, 5, 1, Convention::YIELD);
    REQUIRE(npv1 == Catch::Approx(3992.7100370781));

    // reinvertir anualmente
    double npv2 = npv_from_growth_coupon(1000, 0.05, 0.08, 5, 1, Convention::YIELD);
    REQUIRE(npv2 == Catch::Approx(4379.4737959505));

    // reinvertir mensualmente
    double npv3 = npv_from_growth_coupon(1000, 0.05, 0.08, 5, 1, Convention::YIELD, 12, Convention::YIELD);
    REQUIRE(npv3 == Catch::Approx(4388.9035111952));
}
