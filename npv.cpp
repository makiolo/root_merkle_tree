// https://www.linkedin.com/pulse/python-bootstrapping-zero-curve-sheikh-pancham#:~:text=The%20objective%20of%20bootstrapping%20is,the%20end%20of%20its%20term.

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <catch_amalgamated.hpp>
// header only date: https://raw.githubusercontent.com/HowardHinnant/date/master/include/date/date.h
// date algorithms: http://howardhinnant.github.io/date_algorithms.html
// date doc: https://howardhinnant.github.io/date.html
#include "date.h"

class InterestRate;
class DiscountFactor;

enum Convention
{
    LINEAR,
    YIELD,
    EXPONENTIAL,
};

enum Frequency
{
    ANNUAL = 1,
    SEMIANNUAL = 2,
    QUATERLY = 4,
    MONTHLY = 12,
};

enum DayCountConvention
{
    ACT_ACT,
    ACT_360,
    ACT_365,
    EQUALS,
};

// convert df and zc
double df2zc(double df, double maturity, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double zc2df(double zc, double maturity, Convention conv = Convention::YIELD, int compound_times = Frequency::ANNUAL);
InterestRate equivalent_rate(double rate, Convention conv, int compound_times = Convention::YIELD, Convention other_convention = Convention::YIELD, int other_compound_times = Frequency::ANNUAL);
InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times = Frequency::ANNUAL);
std::vector<DiscountFactor> get_discount_factors_1_T(double r, double years, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
std::vector<DiscountFactor> get_discount_factors_0_T_less1(double r, double years, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// one cash flow
double to_present_value(double cash, const InterestRate& r, double maturity);
double to_future_value(double cash, const InterestRate& r, double maturity);

// real
double real_from_coupon(double coupon, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double coupon_from_real(double real, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// fv - coupon - fv
double npv_from_coupon(double coupon, const InterestRate& interest_rate, double years);
double coupon_from_npv(double npv, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double coupon_from_fv(double fv, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double fv_from_coupon(double coupon, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// growth_coupon
double npv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL, Convention g_convention = Convention::YIELD, int g_compound_times = Frequency::ANNUAL);
double fv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL, Convention g_convention = Convention::YIELD, int g_compound_times = Frequency::ANNUAL);
double coupon_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL, Convention g_convention = Convention::YIELD, int g_compound_times = Frequency::ANNUAL);

/*
growth_coupon_from_npv
growth_coupon_from_coupon
grouth_coupon_from_fv
*/
double compute_irr(const std::vector<double>& cf, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// value products
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, double maturity);
double bond_npv(double face_value, double coupon, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double stock_npv(double investment, double dividend, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double stock_fv(double investment, double dividend, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);
double stock_real(double investment, double dividend, double interest_rate, double maturity, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// calcular r
InterestRate on_capital(double initial, double final, double maturity = 1.0, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);


struct Maturity
{
    date::year_month_day pillar;
    double value;
};

struct Calendar
{
    date::year_month_day& start_date;
    date::year_month_day& end_date;
    int tenor;
    DayCountConvention dc_convention;
};

std::vector<Maturity> generate_pay_calendar(const Calendar& cal, bool begin_mode = true)
{
    using namespace date;
    std::vector<Maturity> data;
    auto d = cal.start_date;
    int i = 0;
    if (!begin_mode)
    {
        d += months(cal.tenor);
        i++;
    }
    while ((begin_mode && (d < cal.end_date)) || (!begin_mode && (d <= cal.end_date)))
    {
        // ACT/ACT
        double actual;
        switch (cal.dc_convention)
        {
        case DayCountConvention::ACT_ACT:
            actual = double((sys_days{ dec / day(31) / d.year() } - sys_days{ jan / day(1) / d.year() }).count());
            break;
        case DayCountConvention::ACT_365:
            actual = 365.0;
            break;
        case DayCountConvention::EQUALS:
            actual = double(i) / (12.0 / cal.tenor);
            break;
        case DayCountConvention::ACT_360:
        default:
            actual = 360.0;
            break;
        }

        if (cal.dc_convention != DayCountConvention::EQUALS)
        {
            double maturity = double((sys_days{ d } - sys_days{ cal.start_date }).count()) / actual;
            data.emplace_back(Maturity{ d, maturity });
        }
        else
        {
            data.emplace_back(Maturity{ d, actual });
        }

        // TODO: use switch tenor
        d += months(cal.tenor);
        i += 1;
    }
    return data;
}

class DiscountFactor
{
public:
    explicit DiscountFactor(double df_)
        : df(df_)
    {

    }

    DiscountFactor(const DiscountFactor& other)
    {
        df = other.df;
    }

    DiscountFactor(DiscountFactor&& other) noexcept {
        df = other.df;
    }

    [[nodiscard]] InterestRate to_interest_rate(double maturity, Convention convention_ = Convention::YIELD, int compound_times_ = Frequency::ANNUAL) const;

public:
    double df;
};

class InterestRate
{
public:
    explicit InterestRate(double interest_rate_, Convention convention_ = Convention::YIELD, int compound_times_ = 1)
        : r(interest_rate_)
        , conv(convention_)
        , c(compound_times_)
    {

    }

    InterestRate(const InterestRate& other)
    {
        r = other.r;
        c = other.c;
        conv = other.conv;
    }

    InterestRate(InterestRate&& other) noexcept
    {
        r = other.r;
        c = other.c;
        conv = other.conv;
    }

    bool operator==(const InterestRate& rhs) const
    {
        return  r == rhs.r &&
            c == rhs.c &&
            conv == rhs.conv;
    }

    [[nodiscard]] std::vector<DiscountFactor> get_discount_factors_to_0(const Calendar& cal) const
    {
        std::vector<DiscountFactor> dfs;
        // for (int i = 1; i <= years * c; ++i)

        for (auto& maturity : generate_pay_calendar(cal, false))
        {
            // dfs.push_back(to_discount_factor(double(i) / c));
            dfs.push_back(to_discount_factor(maturity));
        }
        return dfs;
    }

    [[nodiscard]] std::vector<DiscountFactor> get_discount_factors_to_T(const Calendar& cal) const
    {
        std::vector<DiscountFactor> dfs;
        // for (int i = 0; i < years * c; ++i)
        for (auto& maturity : generate_pay_calendar(cal, true))
        {
            // dfs.push_back(to_discount_factor(double(i) / c));
            dfs.push_back(to_discount_factor(maturity));
        }
        return dfs;
    }

    [[nodiscard]] DiscountFactor to_discount_factor(const Maturity& maturity) const
    {
        return DiscountFactor(zc2df(r, maturity.value, conv, c));
    }

    [[nodiscard]] DiscountFactor to_discount_factor(double maturity) const
    {
        return DiscountFactor(zc2df(r, maturity, conv, c));
    }

    [[nodiscard]] InterestRate to_other_interest_rate(Convention other_convention, int other_compound_times = Frequency::ANNUAL) const
    {
        return equivalent_rate(r, conv, c, other_convention, other_compound_times);
    }

public:
    double r;  // annual rate
    int c;  // reinversions each year
    Convention conv;  // convention
};


// TODO:
class CashFlow
{
public:
    explicit CashFlow(double maturity_, double cash_)
        : maturity(maturity_)
        , cash(cash_)
    {

    }

public:
    double cash;
    double maturity;
};

// TODO:
class Coupon
{
public:
    explicit Coupon(double years_, double cash_)
        : years(years_)
        , cash(cash_)
    {

    }

    // to_npv
    // to_fv

public:
    double cash;
    double years;
    // InterestRate growth;
};

class InitialCashFlow
{
public:
    // to_coupon
    // to_fv
};

class FinalCashFlow
{
public:
    // to_coupon
    // to_npv
};

InterestRate DiscountFactor::to_interest_rate(double maturity, Convention convention_, int compound_times_) const
{
    return InterestRate(df2zc(df, maturity, convention_, compound_times_), convention_, compound_times_);
}

// ********************** //

double df2zc(double df, double maturity, Convention conv, int compound_times)
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

double zc2df(double zc, double maturity, Convention conv, int compound_times)
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

#define LOW_RATE  -0.999
#define HIGH_RATE  0.999
#define MAX_ITERATION 1000
#define PRECISION_REQ 0.00000001
double compute_irr(const std::vector<double>& cf, Convention convention, int compound_times)
{
    int i = 0, j = 0;
    double m = 0.0;
    double old = 0.00;
    double new_ = 0.00;
    double oldguessRate = LOW_RATE;
    double newguessRate = LOW_RATE;
    double guessRate = LOW_RATE;
    double lowGuessRate = LOW_RATE;
    double highGuessRate = HIGH_RATE;
    double npv = 0.0;
    double discount_factor = 0.0;
    for (i = 0; i < MAX_ITERATION; i++)
    {
        npv = 0.00;
        for (j = 0; j < cf.size(); j++)
        {
            discount_factor = zc2df(guessRate, j, convention, compound_times);
            npv = npv + (cf[j] * discount_factor);
        }
        /* Stop checking once the required precision is achieved */
        if ((npv > 0) && (npv < PRECISION_REQ))
            break;
        if (old == 0)
            old = npv;
        else
            old = new_;
        new_ = npv;
        if (i > 0)
        {
            if (old < new_)
            {
                if (old < 0 && new_ < 0)
                    highGuessRate = newguessRate;
                else
                    lowGuessRate = newguessRate;
            }
            else
            {
                if (old > 0 && new_ > 0)
                    lowGuessRate = newguessRate;
                else
                    highGuessRate = newguessRate;
            }
        }
        oldguessRate = guessRate;
        guessRate = (lowGuessRate + highGuessRate) / 2;
        newguessRate = guessRate;
    }
    return guessRate;
}


InterestRate equivalent_rate(double rate, Convention convention, int compound_times, Convention other_convention, int other_compound_times)
{
    return InterestRate(rate, convention, compound_times)
        .to_discount_factor(1.0)
        .to_interest_rate(1.0, other_convention, other_compound_times);
}

InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times)
{
    return equivalent_rate(rate, Convention::YIELD, compound_times, Convention::YIELD, other_compound_times);
}

std::vector<DiscountFactor> get_discount_factors_1_T(double r, const Calendar& cal, Convention convention, int compound_times)
{
    InterestRate ir(r, convention, compound_times);
    return ir.get_discount_factors_to_0(cal);
}

std::vector<DiscountFactor> get_discount_factors_0_T_less1(double r, const Calendar& cal, Convention convention, int compound_times)
{
    InterestRate ir(r, convention, compound_times);
    return ir.get_discount_factors_to_T(cal);
}

// tenemos un cash en "maturity" y nos lo traemos a presente
double to_present_value(double cash, const InterestRate& r, double maturity)
{
    return cash * r.to_discount_factor(maturity).df;
}

// tenemos un cash en "maturity" y nos lo traemos a futuro
double to_future_value(double cash, const InterestRate& r, double maturity)
{
    return cash / r.to_discount_factor(maturity).df;
}

double real_from_coupon(double coupon, double maturity, Convention conv, int compound_times)
{
    return coupon * maturity * compound_times;
}

double coupon_from_real(double real, double maturity, Convention conv, int compound_times)
{
    return real / (maturity * compound_times);
}

// only coupons
double npv_from_coupon(double coupon, const InterestRate& interest_rate, double years)
{
    return npv_from_growth_coupon(coupon, 0.0, interest_rate.r, years, interest_rate.conv, interest_rate.c);
}

double npv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, Convention convention, int compound_times, Convention g_convention, int g_compound_times)
{
    auto start_date = date::year(2012) / 1 / 1;
    auto end_date = date::year(2012 + maturity) / 1 / 1;
    int tenor = int(12.0 / compound_times);
    Calendar cal{start_date, end_date, tenor, DayCountConvention::EQUALS};
    auto dfs = get_discount_factors_1_T(interest_rate, cal, convention, compound_times);
    double npv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        npv += (df.df * (coupon / compound_times)) / zc2df(g, i, g_convention, g_compound_times);
        i += 1.0;
    }
    return npv;
}

double fv_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, Convention convention, int compound_times, Convention g_convention, int g_compound_times)
{
    double npv = npv_from_growth_coupon(coupon, g, interest_rate, maturity, convention, compound_times, g_convention, g_compound_times);
    return to_future_value(npv, InterestRate(interest_rate, convention, compound_times), maturity);
}

double fv_from_coupon(double coupon, double interest_rate, double maturity, Convention convention, int compound_times)
{
    return fv_from_growth_coupon(coupon, 0.0, interest_rate, maturity, convention, compound_times);
}

double coupon_from_npv(double npv, double interest_rate, double maturity, Convention convention, int compound_times)
{
    using namespace date;
    auto start_date = 2012_y / 1 / 1;
    auto end_date = year(2012 + maturity) / 1 / 1;
    int tenor = int(12.0 / compound_times);
    Calendar cal{ start_date, end_date, tenor, DayCountConvention::EQUALS };
    auto dfs = get_discount_factors_1_T(interest_rate, cal, convention, compound_times);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += df.df;
    }
    return (npv * compound_times) / total_df;
}

// VAN = coupons - initial investement
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, double maturity)
{
    return npv_from_coupon(coupon, interest_rate, maturity) - investment;
}

// coupons + payment on yield-maturity
double bond_npv(double face_value, double coupon, double interest_rate, double maturity, Convention convention, int compound_times)
{
    double coupons = npv_from_coupon(coupon, InterestRate(interest_rate, convention, compound_times), maturity);
    double yield_on_payment = to_present_value(face_value, InterestRate(interest_rate, convention, compound_times), maturity);
    return coupons + yield_on_payment;
}

// stock investment = coupons + payment on yield-maturity - initial investement
double stock_npv(double investment, double dividend, double interest_rate, double maturity, Convention convention, int compound_times)
{
    double coupon = investment * (dividend - interest_rate);
    double coupons = npv_from_coupon(coupon, InterestRate(interest_rate, convention, compound_times), maturity);
    double yield_on_payment = to_present_value(investment, InterestRate(interest_rate, convention, compound_times), maturity);
    double npv = coupons + yield_on_payment - investment;
    return npv;
}

// TODO: reimplement it
double stock_fv(double investment, double dividend, double interest_rate, double maturity, Convention convention, int compound_times)
{
    double coupon = investment * (dividend - interest_rate);
    double coupons = fv_from_coupon(coupon, interest_rate, maturity, convention, compound_times);
    double npv = coupons;
    return npv;
}

// TODO: reimplement it
double stock_real(double investment, double dividend, double interest_rate, double maturity, Convention convention, int compound_times)
{
    double coupon = investment * (dividend - interest_rate);
    double investment_fv = to_future_value(investment, InterestRate(interest_rate, convention, compound_times), maturity);
    double coupons = real_from_coupon(coupon, maturity, convention, compound_times);
    double npv = investment - coupons - investment_fv;
    return npv;
}

double coupon_from_growth_coupon(double coupon, double g, double interest_rate, double maturity, Convention convention, int compound_times, Convention g_convention, int g_compound_times)
{
    double npv = npv_from_growth_coupon(coupon, g, interest_rate, maturity, convention, compound_times, g_convention, g_compound_times);
    return coupon_from_npv(npv, interest_rate, maturity, convention, compound_times);
}

double coupon_from_fv(double fv, double interest_rate, double maturity, Convention convention, int compound_times)
{
    using namespace date;
    auto start_date = 2012_y / 1 / 1;
    auto end_date = year(2012 + maturity) / 1 / 1;
    int tenor = int(12.0 / compound_times);
    Calendar cal{ start_date, end_date, tenor, DayCountConvention::EQUALS };
    auto dfs = get_discount_factors_0_T_less1(interest_rate, cal, convention, compound_times);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += 1.0 / df.df;
    }
    return (fv * compound_times) / total_df;
}

InterestRate on_capital(double initial, double final, double maturity, Convention convention, int compound_times)
{
    if (convention == Convention::YIELD && compound_times == Frequency::ANNUAL)
    {
        // cagr
        return InterestRate(pow(final / initial, 1.0 / maturity) - 1.0);
    }
    else
    {
        return InterestRate((final - initial) / initial, Convention::LINEAR)
            .to_discount_factor(1.0)
            .to_interest_rate(maturity, convention, compound_times);
    }
}

TEST_CASE("bond_npv", "[fv]") {

    // Comparado con: https://mathcracker.com/es/calculadora-valor-bonos#results
    // valor presente de un bono
    // valorar un bono que da un yield "seguro" haciendo otros proyectos risk free
    double npv = bond_npv(16000,
        // inversion "segura" ofrecida por el bono
        100, 0.06,
        // inversion libre de riesgo en el mercado
        20);

    REQUIRE(npv == Catch::Approx(6135.87));
}

TEST_CASE("fv_from_coupon", "[fv]") {

    // Ahorro inicial en el futuro

    double initial = 10000;
    double r = 0.07;
    double maturity = 8;
    double fv1 = to_future_value(initial, InterestRate(r), maturity);
    double aportado1 = initial;
    double presente1 = initial;

    REQUIRE(aportado1 == Catch::Approx(10000));
    REQUIRE(presente1 == Catch::Approx(10000));
    REQUIRE(fv1 == Catch::Approx(17181.8617983192));

    // Ahorro periodico (anual)
    double cuota;

    cuota = 5000;
    double aportado2 = real_from_coupon(cuota, maturity, Convention::YIELD);
    double presente2 = npv_from_coupon(cuota, InterestRate(r), maturity);
    double fv2 = fv_from_coupon(cuota, r, maturity, Convention::YIELD);

    REQUIRE(aportado2 == Catch::Approx(40000.0));
    REQUIRE(presente2 == Catch::Approx(29856.4925310687));
    REQUIRE(fv2 == Catch::Approx(51299.0128451372));

    // Ahorro periodico (mensual)

    cuota = 1000;
    double compound_times = 12;
    double aportado3 = real_from_coupon(cuota, maturity, Convention::YIELD, compound_times);
    double presente3 = npv_from_coupon(cuota * compound_times, InterestRate(r, Convention::YIELD, compound_times), maturity);
    double fv3 = fv_from_coupon(cuota * compound_times, r, maturity, Convention::YIELD, compound_times);

    REQUIRE(presente3 == Catch::Approx(73347.5686854354));
    REQUIRE(aportado3 == Catch::Approx(96000.0));
    REQUIRE(fv3 == Catch::Approx(128198.8210340072));

    double final;
    double presente_total;
    double aportado_total = aportado1 + aportado2 + aportado3;
    presente_total = presente1 + presente2 + presente3;
    final = fv1 + fv2 + fv3;
    REQUIRE(coupon_from_real(aportado_total, maturity, Convention::YIELD, 12) == Catch::Approx(1520.8333333333));
    REQUIRE(coupon_from_real(aportado_total, maturity, Convention::YIELD) == Catch::Approx(18250.0));
    REQUIRE(presente_total == Catch::Approx(113204.0612165041));
    REQUIRE(aportado_total == Catch::Approx(146000.0));
    REQUIRE(final == Catch::Approx(196679.6956774635));

    InterestRate r_invest = on_capital(aportado_total, final, maturity);
    REQUIRE(r_invest.r == Catch::Approx(0.0379485678));
}

TEST_CASE("fv_from_coupon2", "[fv]")
{
    // Ahorro periodico (semanal)

    double cuota = 200;
    double frecuencia = 54;
    double maturity = 3.0;
    double r = 0.08;
    double presente = npv_from_coupon(cuota * frecuencia, InterestRate(r), maturity);
    double aportado = real_from_coupon(cuota * frecuencia, maturity, Convention::YIELD);
    double future = fv_from_coupon(cuota * frecuencia, r, maturity, Convention::YIELD);

    REQUIRE(presente == Catch::Approx(27832.6474622771));
    REQUIRE(aportado == Catch::Approx(32400.0));
    REQUIRE(future == Catch::Approx(35061.12));
}

TEST_CASE("cagr_interest", "[fv]") {

    double initial = 11000;
    double final = 21000;
    double past_years = 2;
    double forward_years = 2;
    // past info
    double r = on_capital(initial, final, past_years, Convention::YIELD).r;
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    r = on_capital(initial, final, past_years).r;
    REQUIRE(r * 100 == Catch::Approx(38.1698559416));

    // forward prediction
    REQUIRE(to_future_value(final, InterestRate(r, Convention::YIELD, 12), forward_years) == Catch::Approx(44524.0670913586));

    // trading
    initial = 5000;
    r = 0.10;
    double r_anual = equivalent_rate(r, 12, 1).r;
    double years = 3.0;
    REQUIRE(to_future_value(initial, InterestRate(r, Convention::YIELD, 12), years) == \
        Catch::Approx(to_future_value(initial, InterestRate(r_anual), years)));
}

TEST_CASE("df & zc", "[fv]") {

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::LINEAR, 4), 3, Convention::LINEAR) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::LINEAR), 3, Convention::LINEAR) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::YIELD), 3, Convention::YIELD) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::YIELD), 3, Convention::YIELD) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::YIELD, 4), 3, Convention::YIELD, 4) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::YIELD, 4), 3, Convention::YIELD, 4) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.05));

    REQUIRE(zc2df(df2zc(0.95, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.95));
    REQUIRE(df2zc(zc2df(0.05, 3, Convention::EXPONENTIAL), 3, Convention::EXPONENTIAL) == Catch::Approx(0.05));
}

TEST_CASE("bond_npv2", "[fv]") {

    double cash = 17181.8617983192;
    double r = 0.07;
    double maturity = 8;
    REQUIRE(to_present_value(cash, InterestRate(r), maturity) == Catch::Approx(10000));

    // future value
    double fv = 51299.0128451372;
    REQUIRE(coupon_from_fv(fv, r, maturity, Convention::YIELD) == Catch::Approx(5000));
    REQUIRE(fv_from_coupon(5000, r, maturity, Convention::YIELD) == Catch::Approx(fv));

    // traer flujos futuros a presente
    double npv = npv_from_coupon(5000, InterestRate(r), maturity);
    REQUIRE(npv == Catch::Approx(29856.4925310687));

    // Traerme a presente flujos futuros anuales
    REQUIRE(coupon_from_npv(npv, r, maturity, Convention::YIELD) == Catch::Approx(5000));

    REQUIRE(classic_npv(
        // inversion
        6000,
        // cuota
        500,
        // free risk rate
        InterestRate(0.01),
        // years
        1) == Catch::Approx(-5504.9504950495));

    double npv1 = classic_npv(1000, 100, InterestRate(-0.1940185202), 6);
    REQUIRE(npv1 == Catch::Approx(364.7956282082));

    std::vector<double> cf = { -1000, 100, 100, 100, 100, 100 };
    double irr = compute_irr(cf);
    REQUIRE(irr == Catch::Approx(-0.1940185202));
}

TEST_CASE("real coupon", "[fv]") {

    double coupon_netflix = 9.9;
    double maturity = 10;
    double real = real_from_coupon(coupon_netflix, maturity, Convention::YIELD, 12);

    REQUIRE(real == Catch::Approx(1188.0));

    // dividendo 0.08, precio dinero 0.03

    double npv = stock_npv(1000, 0.08, 0.03, maturity, Convention::YIELD);
    REQUIRE(npv == Catch::Approx(170.6040567355));

    double fv = stock_fv(1000, 0.08, 0.03, maturity, Convention::YIELD);
    REQUIRE(fv == Catch::Approx(573.1939655735));

    double real2 = stock_real(1000, 0.08, 0.03, maturity, Convention::YIELD);
    REQUIRE(real2 == Catch::Approx(-843.9163793441));

    // dividendo 0.08, precio dinero 0.12

    double npv_ = stock_npv(1000, 0.08, 0.12, maturity, Convention::YIELD);
    REQUIRE(npv_ == Catch::Approx(-904.0356845457));

    double fv_ = stock_fv(1000, 0.08, 0.12, maturity, Convention::YIELD);
    REQUIRE(fv_ == Catch::Approx(-701.9494027814));

    double real2_ = stock_real(1000, 0.08, 0.12, maturity, Convention::YIELD);
    REQUIRE(real2_ == Catch::Approx(-1705.8482083442));

    REQUIRE(on_capital(npv, fv, maturity, Convention::EXPONENTIAL).r == Catch::Approx(0.1211878754));
    REQUIRE(on_capital(npv_, fv_, maturity, Convention::EXPONENTIAL).r == Catch::Approx(-0.0253007508));
}

TEST_CASE("tn & te", "[fv]")
{
    double a = 0.05 / 12;
    // TASA NOMINAL a TASA EFECTIVA
    double b = equivalent_rate(0.05, 12, 1).r / 12;
    // TASA EFECTIVA A TASA NOMINAL
    double c = equivalent_rate(0.05, 1, 12).r / 12;

    double c1 = 1000 * a;
    double c2 = 1000 * b;
    double c3 = 1000 * c;

    REQUIRE(c1 == Catch::Approx(4.1666666667));
    REQUIRE(c2 == Catch::Approx(4.2634914901));
    REQUIRE(c3 == Catch::Approx(4.0741237836));

    // 5% reinvirtiendo 1 vez al añao
    REQUIRE(on_capital(1000, 1000 + (c1 * 12)).r == Catch::Approx(0.05));
    // 5% reinvirtiendo 12 veces equivalen a 5.1161% reinvirtiendo 1
    REQUIRE(on_capital(1000, 1000 + (c2 * 12)).r == Catch::Approx(0.0511618979));
    // 5% reinvirtiendo 1 vez equivalen a 4.888% reinvirtiendo 12
    REQUIRE(on_capital(1000, 1000 + (c3 * 12)).r == Catch::Approx(0.0488894854));

    //REQUIRE(tn_2_te(0.05, 12) == Catch::Approx(0.0511618979));
    REQUIRE(equivalent_rate(0.05, 12, 1).r == Catch::Approx(0.0511618979));

    //REQUIRE(te_2_tn(0.05, 12) == Catch::Approx(0.0488894854));
    REQUIRE(equivalent_rate(0.05, 1, 12).r == Catch::Approx(0.0488894854));

    REQUIRE(equivalent_rate(0.0488894854, 12, 1).r == Catch::Approx(0.05));
    REQUIRE(equivalent_rate(0.0511618979, 1, 12).r == Catch::Approx(0.05));

    REQUIRE(equivalent_rate(0.01, 365, 1).r == Catch::Approx(0.0100500287));
    REQUIRE(equivalent_rate(0.01, 1, 365).r == Catch::Approx(0.0099504665));

    /*
    10% mensual con reinversion mensual
    */
    double  fv = to_future_value(1000, InterestRate(0.10 * 12, Convention::YIELD, 12), 1);
    REQUIRE(fv == Catch::Approx(3138.428376721));
    REQUIRE(on_capital(1000, fv).r == Catch::Approx(equivalent_rate(0.10 * 12, 12, 1).r));

    /*
    10% mensual con reinversion anual = 120%
    */
    double  fv2 = to_future_value(1000, InterestRate(0.10 * 12), 1);
    REQUIRE(fv2 == Catch::Approx(2200.0));
    REQUIRE(on_capital(1000, fv2).r == Catch::Approx(equivalent_rate(0.10 * 12, 1, 1).r));

    /*
    2% semanal con reinversion semanal = 191.34%
    */
    double  fv3 = to_future_value(1000, InterestRate(0.02 * 54, Convention::YIELD, 54), 1);
    REQUIRE(fv3 == Catch::Approx(2913.4614441403));
    REQUIRE(on_capital(1000, fv3).r == Catch::Approx(InterestRate(0.02 * 54, Convention::YIELD, 54).to_other_interest_rate(Convention::YIELD).r));

    /*
    2% semanal con reinversion continua = 194.46%
    */
    double  fv4 = to_future_value(1000, InterestRate(0.02 * 54, Convention::EXPONENTIAL), 1);
    REQUIRE(fv4 == Catch::Approx(2944.6795510655));
    // ¿Como calcular ese CAGR?
    REQUIRE(on_capital(1000, fv4).r == Catch::Approx(InterestRate(0.02 * 54, Convention::EXPONENTIAL).to_other_interest_rate(Convention::YIELD).r));

    REQUIRE(equivalent_rate(0.05, 1, 12) == equivalent_rate(0.05, Convention::YIELD, 1, Convention::YIELD, 12));

    InterestRate other_r = InterestRate(0.2).to_other_interest_rate(Convention::EXPONENTIAL);
    REQUIRE(other_r.r == Catch::Approx(0.1823215568));
}

TEST_CASE("coupon growth", "[fv]")
{
    // el dividendo no crece
    double npv1 = npv_from_coupon(1000, InterestRate(0.08), 5);
    REQUIRE(npv1 == Catch::Approx(3992.7100370781));

    // reinvertir anualmente
    double npv2 = npv_from_growth_coupon(1000, 0.05, 0.08, 5);
    REQUIRE(npv2 == Catch::Approx(4379.4737959505));
}

TEST_CASE("coupon growth2", "[fv]")
{
    // npv y fv from growth cupon

    double npv_from_gcoupon = npv_from_growth_coupon(1000, 0.05, 0.08, 5, Convention::YIELD, 12, Convention::YIELD, 12);
    REQUIRE(npv_from_gcoupon == Catch::Approx(23219.4483321569));

    double fv_from_gcoupon = fv_from_growth_coupon(1000, 0.05, 0.08, 5, Convention::YIELD, 12, Convention::YIELD, 12);
    REQUIRE(fv_from_gcoupon == Catch::Approx(34593.3954467948));


    // cupon from growth cupon

    double fixed_coupon = coupon_from_growth_coupon(1000, 0.05, 0.08, 5, Convention::YIELD, 12, Convention::YIELD, 12);
    REQUIRE(fixed_coupon == Catch::Approx(5649.6802745071));

    // fv

    double coupon1 = coupon_from_fv(fv_from_gcoupon, 0.08, 5, Convention::YIELD, 12);
    REQUIRE(coupon1 == Catch::Approx(fixed_coupon));

    double fv4 = fv_from_coupon(coupon1, 0.08, 5, Convention::YIELD, 12);
    REQUIRE(fv4 == Catch::Approx(fv_from_gcoupon));

    double fv5 = fv_from_coupon(fixed_coupon, 0.08, 5, Convention::YIELD, 12);
    REQUIRE(fv5 == Catch::Approx(fv_from_gcoupon));

    // npv

    double coupon2 = coupon_from_npv(npv_from_gcoupon, 0.08, 5, Convention::YIELD, 12);
    REQUIRE(coupon2 == Catch::Approx(fixed_coupon));

    double npv4 = npv_from_coupon(coupon2, InterestRate(0.08, Convention::YIELD, 12), 5);
    REQUIRE(npv4 == Catch::Approx(npv_from_gcoupon));

    double npv5 = npv_from_coupon(fixed_coupon, InterestRate(0.08, Convention::YIELD, 12), 5);
    REQUIRE(npv5 == Catch::Approx(npv_from_gcoupon));
}

TEST_CASE("date C++20", "[date]")
{
    using namespace date;
    auto x = 2012_y / 1 / 24;
    auto y = 2013_y / 1 / 8;
    auto diff = (sys_days{ y } - sys_days{ x }).count();
    REQUIRE(diff == Catch::Approx(350));

    auto start_date = jan / day(1) / 2020;
    auto end_date = jan / day(1) / 2030;
    double last_maturity;
    for (auto d = start_date; d < end_date; d += months(1))
    {
        // ACT/ACT
        int actual = (sys_days{ dec / day(31) / d.year() } - sys_days{ jan / day(1) / d.year() }).count();
        double maturity = double((sys_days{ d } - sys_days{ start_date }).count()) / double(actual);
        std::cout << maturity << std::endl;
        std::cout << d << ": " << to_present_value(1000, InterestRate(0.05), maturity) << std::endl;
        last_maturity = maturity;
    }
    REQUIRE(last_maturity == Catch::Approx(9.9505494505));
    
    Calendar cal{start_date, end_date, 1, DayCountConvention::EQUALS};

    for (auto& maturity : generate_pay_calendar(cal, true))
    {
        std::cout << "pillar: " << maturity.pillar << " - value: " << maturity.value << std::endl;
    }
    int c = 12;
    for (int i = 0; i < 10 * c; ++i)
    {
        std::cout << "value: " << double(i) / c << std::endl;
    }
}
