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
class Maturity;
class Calendar;
class InterestRate;

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
InterestRate equivalent_rate(double rate, Convention conv, int compound_times, Convention other_convention = Convention::YIELD, int other_compound_times = Frequency::ANNUAL);
InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times = Frequency::ANNUAL);

// one cash flow
double to_present_value(double cash, const InterestRate& r, const Maturity& maturity);
double to_future_value(double cash, const InterestRate& r, const Maturity& maturity);
double to_present_value(double cash, const InterestRate& r, const Calendar& cal);
double to_future_value(double cash, const InterestRate& r, const Calendar& cal);

// fv - coupon - fv
double npv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal);
double coupon_from_npv(double npv, const InterestRate& interest_rate, const Calendar& cal);
double coupon_from_fv(double fv, const InterestRate& interest_rate, const Calendar& cal);
double fv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal);

// growth_coupon
double npv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal);
double fv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal);
double coupon_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal);

/*
growth_coupon_from_npv
growth_coupon_from_coupon
grouth_coupon_from_fv
*/
double compute_irr(const std::vector<double>& cf, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);

// value products
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, const Calendar& cal);
double bond_npv(double face_value, double coupon, const InterestRate& interest_rate, const Calendar& cal);
double stock_npv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal);
double stock_fv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal);

// calcular r
InterestRate on_capital(double initial, double final_value, double maturity = 1.0, Convention convention = Convention::YIELD, int compound_times = Frequency::ANNUAL);


struct Maturity
{
    Maturity(double value_)
        : value(value_)
    {
        
    }

    explicit Maturity(const date::year_month_day& pillar_, double value_)
        : pillar(pillar_)
        , value(value_)
    {

    }

    date::year_month_day pillar;
    double value;
};

struct Calendar
{
    date::year_month_day start_date;
    date::year_month_day end_date;
    int tenor;
    DayCountConvention dc_convention;

    Calendar(const date::year_month_day& start_date_, const date::year_month_day& end_date_, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
        : start_date(start_date_)
        , end_date(end_date_)
        , tenor(tenor_)
        , dc_convention(dc_convention_)
    {
        
    }

    Calendar(const date::year_month_day& start_date_, int duration, int tenor_ = 12, DayCountConvention dc_convention_ = DayCountConvention::EQUALS)
        : start_date(start_date_)
        , end_date(start_date_ + date::years(duration))
        , tenor(tenor_)
        , dc_convention(dc_convention_)
    {

    }

    std::vector<Maturity> generate(bool start = true) const
    {
        using namespace date;

        std::vector<Maturity> data;
        auto pillar_day = start_date;
        int i = 0;
        if (!start)
        {
            pillar_day += months(tenor);
            i++;
        }
        while ((start && (pillar_day < end_date)) || (!start && (pillar_day <= end_date)))
        {
            double count;
            switch (dc_convention)
            {
                case DayCountConvention::ACT_ACT:
                {
                    double m = double((sys_days{ jan / day(1) / (pillar_day.year() + years(1)) } - sys_days{ jan / day(1) / pillar_day.year() }).count());
                    count = double((sys_days{ pillar_day } - sys_days{ start_date }).count()) / m;
                    break;
                }
                case DayCountConvention::ACT_365:
                {
                    count = double((sys_days{ pillar_day } - sys_days{ start_date }).count()) / 365.0;
                    break;
                }
                case DayCountConvention::EQUALS:
                {
                    count = double(i) / (12.0 / tenor);
                    break;
                }
                case DayCountConvention::ACT_360:
                default:
                {
                    count = double((sys_days{ pillar_day } - sys_days{ start_date }).count()) / 360.0;
                    break;
                }
            }
            data.emplace_back(Maturity{ pillar_day, count });
            pillar_day += months(tenor);
            i += 1;
        }
        return data;
    }
};

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

    DiscountFactor(DiscountFactor&& other) noexcept
    {
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

    [[nodiscard]] std::vector<DiscountFactor> get_discount_factors_1_T(const Calendar& cal) const
    {
        std::vector<DiscountFactor> dfs;
        for (auto& maturity : cal.generate(false))
        {
            dfs.push_back(to_discount_factor(maturity));
        }
        return dfs;
    }

    [[nodiscard]] std::vector<DiscountFactor> get_discount_factors_0_T_less1(const Calendar& cal) const
    {
        std::vector<DiscountFactor> dfs;
        for (auto& maturity : cal.generate(true))
        {
            dfs.push_back(to_discount_factor(maturity));
        }
        return dfs;
    }

    [[nodiscard]] Maturity get_first_maturity(const Calendar& cal) const
    {
        auto mats = cal.generate(true);
        return mats.front();
    }

    [[nodiscard]] Maturity get_last_maturity(const Calendar& cal) const
    {
        auto mats = cal.generate(false);
        return mats.back();
    }

    [[nodiscard]] DiscountFactor to_discount_factor(const Maturity& maturity) const
    {
        return DiscountFactor(zc2df(r, maturity.value, conv, c));
    }

    [[nodiscard]] InterestRate to_other_interest_rate(Convention other_convention, int other_compound_times = Frequency::ANNUAL) const
    {
        return equivalent_rate(r, conv, c, other_convention, other_compound_times);
    }

    DiscountFactor direct_discount(const Maturity& one, const Maturity& other);
    InterestRate forward_rate(const Maturity& one, const Maturity& other);
    DiscountFactor next_discount(const Maturity& one, const InterestRate& forward_rate, double m = 1.0);

public:
    double r;  // annual rate
    int c;  // reinversions each year
    Convention conv;  // convention
};

DiscountFactor InterestRate::direct_discount(const Maturity& one, const Maturity& other)
{
    double df0 = to_discount_factor(one).df;
    double df1 = to_discount_factor(other).df;

    return DiscountFactor(df1 / df0);
}

InterestRate InterestRate::forward_rate(const Maturity& one, const Maturity& other)
{
    double df0 = to_discount_factor(one).df;
    double df1 = to_discount_factor(other).df;

    double m = other.value - one.value;

    return InterestRate((df0 / df1 - 1.0) / m);
}

DiscountFactor InterestRate::next_discount(const Maturity& one, const InterestRate& forward_rate, double m)
{
    double discount = to_discount_factor(one).df;

    return DiscountFactor(discount / (1.0 + m * forward_rate.r));
}

// TODO:
class CashFlow
{
public:
    explicit CashFlow(Maturity maturity_, double cash_)
        : maturity(maturity_)
        , cash(cash_)
    {

    }

public:
    Maturity maturity;
    double cash;
};

struct Legs
{
    std::vector<CashFlow> flows;
};

struct Product
{
    Calendar cal;
    std::vector<Legs> legs;
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
        .to_discount_factor(Maturity(1.0))
        .to_interest_rate(1.0, other_convention, other_compound_times);
}

InterestRate equivalent_rate(double rate, int compound_times, int other_compound_times)
{
    return equivalent_rate(rate, Convention::YIELD, compound_times, Convention::YIELD, other_compound_times);
}

/*
std::vector<DiscountFactor> get_discount_factors_1_T(double r, const Calendar& cal, Convention convention, int compound_times)
{
    InterestRate ir(r, convention, compound_times);
    return ir.get_discount_factors_1_T(cal);
}
*/

/*
std::vector<DiscountFactor> get_discount_factors_0_T_less1(double r, const Calendar& cal, Convention convention, int compound_times)
{
    InterestRate ir(r, convention, compound_times);
    return ir.get_discount_factors_0_T_less1(cal);
}
*/

// tenemos un cash en "maturity" y nos lo traemos a "0"
double to_present_value(double cash, const InterestRate& r, const Maturity& maturity)
{
    return cash * r.to_discount_factor(maturity).df;
}

// tenemos un cash en "0" y nos lo traemos a "maturity"
double to_future_value(double cash, const InterestRate& r, const Maturity& maturity)
{
    return cash / r.to_discount_factor(maturity).df;
}

// tenemos un cash al final del calendario y nos lo traemos a "0"
double to_present_value(double cash, const InterestRate& r, const Calendar& cal)
{
    auto maturity = r.get_last_maturity(cal); // obtener maturity del cash (teniendo cierto "cal")
    return to_present_value(cash, r, maturity);
}

// tenemos un en 0 y nos lo traemos al final del calendario
double to_future_value(double cash, const InterestRate& r, const Calendar& cal)
{
    auto maturity = r.get_last_maturity(cal); // obtener maturity del cash (teniendo cierto "cal")
    return to_future_value(cash, r, maturity);
}

/*
double real_from_coupon(double coupon, double maturity, Convention conv, int compound_times)
{
    return coupon * maturity * compound_times;
}

double coupon_from_real(double real, double maturity, Convention conv, int compound_times)
{
    return real / (maturity * compound_times);
}
*/

// only coupons
double npv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    return npv_from_growth_coupon(coupon, InterestRate(0.0), interest_rate, cal);
}

double npv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal)
{
    /*
    auto start_date = date::year(2012) / 1 / 1;
    auto end_date = date::year(2012 + maturity) / 1 / 1;
    int tenor = int(12.0 / compound_times);
    Calendar cal(start_date, end_date, tenor, DayCountConvention::EQUALS);
    */

    // InterestRate ir(interest_rate, convention, compound_times);
    auto dfs = interest_rate.get_discount_factors_1_T(cal);

    // auto dfs = get_discount_factors_1_T(interest_rate, cal, convention, compound_times);
    double npv = 0.0;
    double i = 0.0;
    for (const auto& df : dfs)
    {
        npv += (df.df * (coupon / interest_rate.c)) / zc2df(growth_rate.r, i, growth_rate.conv, growth_rate.c);
        i += 1.0;
    }
    return npv;
}

double fv_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal)
{
    double npv = npv_from_growth_coupon(coupon, growth_rate, interest_rate, cal);
    return to_future_value(npv, interest_rate, cal);
}

double fv_from_coupon(double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    return fv_from_growth_coupon(coupon, InterestRate(0.0), interest_rate, cal);
}

double coupon_from_npv(double npv, const InterestRate& interest_rate, const Calendar& cal)
{
    using namespace date;
    /*
    auto start_date = 2012_y / 1 / 1;
    auto end_date = year(2012 + maturity) / 1 / 1;
    int tenor = int(12.0 / compound_times);
    Calendar cal( start_date, end_date, tenor, DayCountConvention::EQUALS );
    */

    auto dfs = interest_rate.get_discount_factors_1_T(cal);

    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += df.df;
    }
    return (npv * interest_rate.c) / total_df;
}

// VAN = coupons - initial investement
double classic_npv(double investment, double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    return npv_from_coupon(coupon, interest_rate, cal) - investment;
}

// coupons + payment on yield-maturity
double bond_npv(double face_value, double coupon, const InterestRate& interest_rate, const Calendar& cal)
{
    double coupons = npv_from_coupon(coupon, interest_rate, cal);
    double yield_on_payment = to_present_value(face_value, interest_rate, cal);
    return coupons + yield_on_payment;
}

// stock investment = coupons + payment on yield-maturity - initial investement
double stock_npv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal)
{
    double coupon = investment * (dividend - interest_rate.r);
    double coupons = npv_from_coupon(coupon, interest_rate, cal);
    double yield_on_payment = to_present_value(investment, interest_rate, cal);
    double npv = coupons + yield_on_payment - investment;
    return npv;
}

// TODO: reimplement it
double stock_fv(double investment, double dividend, const InterestRate& interest_rate, const Calendar& cal)
{
    double coupon = investment * (dividend - interest_rate.r);
    double coupons = fv_from_coupon(coupon, interest_rate, cal);
    double npv = coupons;
    return npv;
}

double coupon_from_growth_coupon(double coupon, const InterestRate& growth_rate, const InterestRate& interest_rate, const Calendar& cal)
{
    double npv = npv_from_growth_coupon(coupon, growth_rate, interest_rate, cal);
    return coupon_from_npv(npv, interest_rate, cal);
}

double coupon_from_fv(double fv, const InterestRate& interest_rate, const Calendar& cal)
{
    /*
    using namespace date;
    auto start_date = 2012_y / 1 / 1;
    auto end_date = year(2012 + maturity) / 1 / 1;
    int tenor = int(12.0 / compound_times);
    Calendar cal( start_date, end_date, tenor, DayCountConvention::EQUALS );
    */

    // InterestRate ir(interest_rate, convention, compound_times);
    auto dfs = interest_rate.get_discount_factors_0_T_less1(cal);

    // auto dfs = get_discount_factors_0_T_less1(interest_rate, cal, convention, compound_times);
    double total_df = 0.0;
    for (const auto& df : dfs)
    {
        total_df += 1.0 / df.df;
    }
    return (fv * interest_rate.c) / total_df;
}

InterestRate on_capital(double initial, double final_value, double maturity, Convention convention, int compound_times)
{
    if (convention == Convention::YIELD && compound_times == Frequency::ANNUAL)
    {
        // cagr
        return InterestRate(pow(final_value / initial, 1.0 / maturity) - 1.0);
    }
    else
    {
        return InterestRate((final_value - initial) / initial, Convention::LINEAR)
            .to_discount_factor(Maturity(1.0))
            .to_interest_rate(maturity, convention, compound_times);
    }
}

TEST_CASE("bond_npv", "[fv]") {

    using namespace date;

    // Comparado con: https://mathcracker.com/es/calculadora-valor-bonos#results
    // valor presente de un bono
    // valorar un bono que da un yield "seguro" haciendo otros proyectos risk free
    double npv = bond_npv(
        // face value
        16000,
        // cupon
        100, 
        InterestRate(0.06),
        // calendar
        Calendar(2022_y/1/1, 20));

    REQUIRE(npv == Catch::Approx(6135.87));
}

TEST_CASE("fv_from_coupon", "[fv]") {

    using namespace date;

    // Ahorro inicial en el futuro

    double initial = 10000;
    double r = 0.07;
    double maturity = 8;
    auto cal = Calendar(2022_y / 1 / 1, maturity);
    double fv1 = to_future_value(initial, InterestRate(r), cal);
    double aportado1 = initial;
    double presente1 = initial;

    REQUIRE(aportado1 == Catch::Approx(10000));
    REQUIRE(presente1 == Catch::Approx(10000));
    REQUIRE(fv1 == Catch::Approx(17181.8617983192));

    // Ahorro periodico (anual)
    double cuota;

    cuota = 5000;
    //double aportado2 = real_from_coupon(cuota, maturity, Convention::YIELD);
    double presente2 = npv_from_coupon(cuota, InterestRate(r), cal);
    double fv2 = fv_from_coupon(cuota, InterestRate(r), cal);

    // REQUIRE(aportado2 == Catch::Approx(40000.0));
    REQUIRE(presente2 == Catch::Approx(29856.4925310687));
    REQUIRE(fv2 == Catch::Approx(51299.0128451372));

    // Ahorro periodico (mensual)

    cuota = 1000;
    double compound_times = 12;
    auto cal2 = Calendar(2022_y / 1 / 1, maturity, 1);
    //double aportado3 = real_from_coupon(cuota, maturity, Convention::YIELD, compound_times);
    double presente3 = npv_from_coupon(cuota * compound_times, InterestRate(r, Convention::YIELD, compound_times), cal2);
    double fv3 = fv_from_coupon(cuota * compound_times, InterestRate(r, Convention::YIELD, compound_times), cal2);

    REQUIRE(presente3 == Catch::Approx(73347.5686854354));
    // REQUIRE(aportado3 == Catch::Approx(96000.0));
    REQUIRE(fv3 == Catch::Approx(128198.8210340072));

    double final_value;
    double presente_total;
    double aportado_total = aportado1; // + aportado2 + aportado3;
    presente_total = presente1 + presente2 + presente3;
    final_value = fv1 + fv2 + fv3;
    //REQUIRE(coupon_from_real(aportado_total, maturity, Convention::YIELD, 12) == Catch::Approx(1520.8333333333));
    //REQUIRE(coupon_from_real(aportado_total, maturity, Convention::YIELD) == Catch::Approx(18250.0));
    REQUIRE(presente_total == Catch::Approx(113204.0612165041));
    REQUIRE(aportado_total == Catch::Approx(10000));
    REQUIRE(final_value == Catch::Approx(196679.6956774635));

    InterestRate r_invest = on_capital(aportado_total, final_value, maturity);
    REQUIRE(r_invest.r == Catch::Approx(0.4511755111));
}

TEST_CASE("fv_from_coupon2", "[fv]")
{
    // Ahorro periodico (semanal)
    using namespace date;

    double cuota = 200;
    double frecuencia = 54;
    double maturity = 3.0;
    double r = 0.08;
    auto cal = Calendar(2022_y / 1 / 1, maturity);
    double presente = npv_from_coupon(cuota * frecuencia, InterestRate(r), cal);
    //double aportado = real_from_coupon(cuota * frecuencia, maturity, Convention::YIELD);
    double future = fv_from_coupon(cuota * frecuencia, InterestRate(r), cal);

    REQUIRE(presente == Catch::Approx(27832.6474622771));
    //REQUIRE(aportado == Catch::Approx(32400.0));
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

    using namespace date;

    double cash = 17181.8617983192;
    double r = 0.07;
    double maturity = 8;
    auto cal = Calendar(2022_y / 1 / 1, maturity);
    REQUIRE(to_present_value(cash, InterestRate(r), cal) == Catch::Approx(10000));

    // future value
    double fv = 51299.0128451372;
    REQUIRE(coupon_from_fv(fv, InterestRate(r), cal) == Catch::Approx(5000));
    REQUIRE(fv_from_coupon(5000, InterestRate(r), cal) == Catch::Approx(fv));

    // traer flujos futuros a presente
    double npv = npv_from_coupon(5000, InterestRate(r), cal);
    REQUIRE(npv == Catch::Approx(29856.4925310687));

    // Traerme a presente flujos futuros anuales
    REQUIRE(coupon_from_npv(npv, InterestRate(r), cal) == Catch::Approx(5000));

    REQUIRE(classic_npv(
        // inversion
        6000,
        // cuota
        500,
        // free risk rate
        InterestRate(0.01),
        // years
        Calendar(2022_y / 1 / 1, 1)) == Catch::Approx(-5504.9504950495));

    double npv1 = classic_npv(1000, 100, InterestRate(-0.1940185202), Calendar(2022_y / 1 / 1, 6));
    REQUIRE(npv1 == Catch::Approx(364.7956282082));

    std::vector<double> cf = { -1000, 100, 100, 100, 100, 100 };
    double irr = compute_irr(cf);
    REQUIRE(irr == Catch::Approx(-0.1940185202));
}

TEST_CASE("real coupon", "[fv]") {

    using namespace date;

    double coupon_netflix = 9.9;
    double maturity = 10;

    auto cal = Calendar(2022_y / 1 / 1, maturity);

    //double real = real_from_coupon(coupon_netflix, maturity, Convention::YIELD, 12);
    //REQUIRE(real == Catch::Approx(1188.0));

    // dividendo 0.08, precio dinero 0.03

    double npv = stock_npv(1000, 0.08, InterestRate(0.03), cal);
    REQUIRE(npv == Catch::Approx(170.6040567355));

    double fv = stock_fv(1000, 0.08, InterestRate(0.03), cal);
    REQUIRE(fv == Catch::Approx(573.1939655735));

    // double real2 = stock_real(1000, 0.08, 0.03, maturity, Convention::YIELD);
    // REQUIRE(real2 == Catch::Approx(-843.9163793441));

    // dividendo 0.08, precio dinero 0.12

    double npv_ = stock_npv(1000, 0.08, InterestRate(0.12), cal);
    REQUIRE(npv_ == Catch::Approx(-904.0356845457));

    double fv_ = stock_fv(1000, 0.08, InterestRate(0.12), cal);
    REQUIRE(fv_ == Catch::Approx(-701.9494027814));

    // double real2_ = stock_real(1000, 0.08, 0.12, maturity, Convention::YIELD);
    // REQUIRE(real2_ == Catch::Approx(-1705.8482083442));

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
    using namespace date;

    auto cal = Calendar(2022_y / 1 / 1, 5);

    // el dividendo no crece
    double npv1 = npv_from_coupon(1000, InterestRate(0.08), cal);
    REQUIRE(npv1 == Catch::Approx(3992.7100370781));

    // reinvertir anualmente
    double npv2 = npv_from_growth_coupon(1000, InterestRate(0.05), InterestRate(0.08), cal);
    REQUIRE(npv2 == Catch::Approx(4379.4737959505));
}

TEST_CASE("coupon growth2", "[fv]")
{
    using namespace date;

    auto cal = Calendar(2022_y / 1 / 1, 5, 1);

    // npv y fv from growth cupon

    double npv_from_gcoupon = npv_from_growth_coupon(1000, InterestRate(0.05, Convention::YIELD, 12), InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(npv_from_gcoupon == Catch::Approx(23219.4483321569));

    double fv_from_gcoupon = fv_from_growth_coupon(1000, InterestRate(0.05, Convention::YIELD, 12), InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(fv_from_gcoupon == Catch::Approx(34593.3954467948));


    // cupon from growth cupon

    double fixed_coupon = coupon_from_growth_coupon(1000, 
        InterestRate(0.05, Convention::YIELD, 12), 
        InterestRate(0.08, Convention::YIELD, 12), 
        cal);
    REQUIRE(fixed_coupon == Catch::Approx(5649.6802745071));

    // fv

    double coupon1 = coupon_from_fv(fv_from_gcoupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(coupon1 == Catch::Approx(fixed_coupon));

    double fv4 = fv_from_coupon(coupon1, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(fv4 == Catch::Approx(fv_from_gcoupon));

    double fv5 = fv_from_coupon(fixed_coupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(fv5 == Catch::Approx(fv_from_gcoupon));

    // npv

    double coupon2 = coupon_from_npv(npv_from_gcoupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(coupon2 == Catch::Approx(fixed_coupon));

    double npv4 = npv_from_coupon(coupon2, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(npv4 == Catch::Approx(npv_from_gcoupon));

    double npv5 = npv_from_coupon(fixed_coupon, InterestRate(0.08, Convention::YIELD, 12), cal);
    REQUIRE(npv5 == Catch::Approx(npv_from_gcoupon));
}

TEST_CASE("date C++20", "[date]")
{
    using namespace date;
    auto x = 2012_y / 1 / 24;
    auto y = 2013_y / 1 / 8;
    auto diff = (sys_days{ y } - sys_days{ x }).count();
    REQUIRE(diff == Catch::Approx(350));

    auto start_date = jan / last / 2020;
    auto end_date = jan / last / 2030;
    double last_maturity;
    for (auto d = start_date; d < end_date; d += months(1))
    {
        // ACT/ACT
        int actual = (sys_days{ jan / day(1) / (d.year() + years(1)) } - sys_days{ jan / day(1) / d.year() }).count();
        double maturity = double((sys_days{ d } - sys_days{ start_date }).count()) / double(actual);
        std::cout << maturity << std::endl;
        std::cout << "for: " << d << " (dia " << d.day() << ")" << ": " << to_present_value(1000, InterestRate(0.05), maturity) << std::endl;
        last_maturity = maturity;
    }
    REQUIRE(last_maturity == Catch::Approx(9.9232876712));
    
    Calendar cal{start_date, end_date, 3, DayCountConvention::EQUALS };

    std::vector<double> v1, v2;
    for (auto& maturity : cal.generate(true))
    {
        std::cout << "begin mode = true, pillar: " << maturity.pillar << " - value: " << maturity.value << std::endl;
        v1.push_back(maturity.value);
    }
    for (auto& maturity : cal.generate(false))
    {
        std::cout << "begin mode = false, pillar: " << maturity.pillar << " - value: " << maturity.value << std::endl;
        v2.push_back(maturity.value);
    }
    std::vector<double> m1, m2;
    int c = 4;
    for (int i = 0; i < 10 * c; ++i)
    {
        std::cout << "value: " << double(i) / c << std::endl;
        m1.push_back(double(i) / c);
    }
    for (int i = 1; i <= 10 * c; ++i)
    {
        std::cout << "value: " << double(i) / c << std::endl;
        m2.push_back(double(i) / c);
    }
    REQUIRE(v1 == m1);
    REQUIRE(v2 == m2);
}

TEST_CASE("forwards1", "[fw]")
{
    using namespace date;

    auto start_date = jan / day(1) / 2020;
    auto end_date = jan / day(1) / 2030;

    Calendar cal{ start_date, end_date, 12, DayCountConvention::EQUALS };
    auto fixings = cal.generate(true);

    InterestRate r(0.08);

    double cash = 1;

    double fwd2 = r.direct_discount(fixings[0], fixings[1]).df;
    double fwd3 = r.direct_discount(fixings[1], fixings[2]).df;
    double fwd4 = r.direct_discount(fixings[0], fixings[2]).df;

    REQUIRE((fwd2) == Catch::Approx(0.9259259259));
    REQUIRE((fwd3) == Catch::Approx(0.9259259259));
    REQUIRE((fwd4) == Catch::Approx(0.9259259259 * 0.9259259259));

    double fwr1 = r.forward_rate(fixings[0], fixings[1]).to_other_interest_rate(Convention::YIELD, 12).r / 12.0;
    double fwr2 = r.forward_rate(fixings[1], fixings[2]).r;
    double fwr3 = r.forward_rate(fixings[2], fixings[3]).r;

    REQUIRE((fwr1) == Catch::Approx(0.0064340301));
    REQUIRE((fwr2) == Catch::Approx(0.08));
    REQUIRE((fwr3) == Catch::Approx(0.08));

    REQUIRE(to_future_value(cash, r, fixings[0]) == Catch::Approx(1.0));
    REQUIRE(to_future_value(cash, r, fixings[1]) == Catch::Approx(1.08));
    REQUIRE(to_future_value(cash, r, fixings[2]) == Catch::Approx(1.1664));
    REQUIRE(to_future_value(cash, r, cal) == Catch::Approx(2.1589249972728));

    REQUIRE((1.08 * fwd2) == Catch::Approx(1.0));
    REQUIRE((1.1664 * fwd3) == Catch::Approx(1.08));
    REQUIRE((1.259712 * fwd2 * fwd3) == Catch::Approx(1.08));

    REQUIRE(to_present_value(1.0, r, fixings[0]) == Catch::Approx(1.0));
    REQUIRE(to_present_value(1.08, r, fixings[1]) == Catch::Approx(1.0));
    REQUIRE(to_present_value(1.1664, r, fixings[2]) == Catch::Approx(1.0));
    REQUIRE(to_present_value(2.1589249972728, r, cal) == Catch::Approx(1.0));

    double df0 = r.to_discount_factor(fixings[0]).df;
    double df1 = r.to_discount_factor(fixings[1]).df;
    double df2 = r.to_discount_factor(fixings[2]).df;

    REQUIRE(df0 == Catch::Approx(1.0));
    REQUIRE(df1 == Catch::Approx(0.9259259259));
    REQUIRE(df2 == Catch::Approx(0.8573388203));

    REQUIRE(r.next_discount(fixings[1], InterestRate(fwr2), 1.0).df == Catch::Approx(r.direct_discount(fixings[0], fixings[2]).df));
    REQUIRE(r.next_discount(fixings[1], InterestRate(fwr2), 1.0).to_interest_rate(fixings[1].value).r == Catch::Approx(0.1664));

    REQUIRE(r.direct_discount(fixings[0], fixings[1]).to_interest_rate(fixings[1].value).r == Catch::Approx(0.08));
    REQUIRE(r.direct_discount(fixings[1], fixings[2]).to_interest_rate(fixings[1].value).r == Catch::Approx(0.08));
    REQUIRE(r.direct_discount(fixings[0], fixings[2]).to_interest_rate(fixings[1].value).r == Catch::Approx(0.1664));
}
