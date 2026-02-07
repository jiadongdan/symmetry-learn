# scripts/run_tests.py complete fixed version
import sys
import os
import argparse
import unittest

# Check if coverage is available
try:
    import coverage

    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False
    coverage = None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Verbosity level: -v, -vv, -vvv')
    parser.add_argument('--pattern', default='test_*.py',
                        help='Test file pattern (default: test_*.py)')
    parser.add_argument('--test-dir', default='tests',
                        help='Test directory (default: tests)')
    parser.add_argument('--coverage', action='store_true',
                        help='Generate coverage report')
    parser.add_argument('--failfast', action='store_true',
                        help='Stop on first failure')
    parser.add_argument('--buffer', action='store_true',
                        help='Buffer output, show only when test fails')

    return parser.parse_args()


def run_unittest_tests(args):
    """Run unittest tests"""
    # Set verbosity level
    if args.verbose == 0:
        verbosity = 1
    elif args.verbose == 1:
        verbosity = 2
    else:
        verbosity = 3

    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"❌ Error: Test directory '{args.test_dir}' does not exist")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)

    # Discover and load tests
    loader = unittest.TestLoader()
    test_suite = loader.discover(
        start_dir=args.test_dir,
        pattern=args.pattern,
        top_level_dir=os.path.dirname(args.test_dir) or '.'
    )

    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast,
        buffer=args.buffer
    )

    # Run tests
    result = runner.run(test_suite)

    return result


def run_with_coverage(args):
    """Run tests and generate coverage report"""
    if not COVERAGE_AVAILABLE:
        print("❌ Error: coverage module not installed")
        print("    In CI/CD environments, this should already be installed.")
        print("    Please ensure requirements-dev.txt includes coverage")
        sys.exit(1)

    # Create coverage object
    # Note: source parameter needs to be adjusted based on your project structure
    # If your code is in src/symmetry_learn/ directory
    cov = coverage.Coverage(
        source=['src'],  # Measure coverage of all code in src directory
        omit=['*test*', '*__pycache__*', '*/tests/*'],
        branch=True
    )

    cov.start()

    # Run tests
    result = run_unittest_tests(args)

    cov.stop()
    cov.save()

    # Generate report
    print("\n" + "=" * 60)
    print("Coverage Report")
    print("=" * 60)

    # Terminal report
    cov.report(show_missing=True, skip_covered=False)

    # HTML report
    html_dir = 'coverage_report'
    cov.html_report(directory=html_dir)

    # XML report (for CI/CD integration)
    cov.xml_report(outfile='coverage.xml')

    print(f"\n📄 HTML report: file://{os.path.abspath(html_dir)}/index.html")

    return result


def main():
    """Main function"""
    args = parse_args()

    try:
        if args.coverage:
            if not COVERAGE_AVAILABLE:
                print("❌ Error: coverage module not installed, cannot generate coverage report")
                print("    Will run tests without coverage")
                result = run_unittest_tests(args)
            else:
                print("Running tests (with coverage)...")
                result = run_with_coverage(args)
        else:
            print("Running tests...")
            result = run_unittest_tests(args)

        # Exit based on test results
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print(f"\n❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            sys.exit(1)

    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()