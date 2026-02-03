import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Navbar } from './navbar';

describe('Navbar', () => {
  let component: Navbar;
  let fixture: ComponentFixture<Navbar>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Navbar]
    }).compileComponents();

    fixture = TestBed.createComponent(Navbar);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('Navigation Structure', () => {
    it('should have a nav element for screen readers and semantic markup', () => {
      const navElement = fixture.nativeElement.querySelector('nav');
      expect(navElement).toBeTruthy();
    });
  });

  describe('Main Link (py_challenge)', () => {
    it('should display the py_challenge link text', () => {
      const mainLink = fixture.nativeElement.querySelector('a[href="https://github.com/jfaldanam/py_challenge"]');
      expect(mainLink).toBeTruthy();
      expect(mainLink.textContent).toContain('py_challenge');
    });

    it('should link to the py_challenge GitHub repository', () => {
      const mainLink = fixture.nativeElement.querySelector('a[href="https://github.com/jfaldanam/py_challenge"]');
      expect(mainLink.getAttribute('href')).toBe('https://github.com/jfaldanam/py_challenge');
    });

    it('should display the favicon image', () => {
      const mainLink = fixture.nativeElement.querySelector('a[href="https://github.com/jfaldanam/py_challenge"]');
      const img = mainLink.querySelector('img');
      expect(img).toBeTruthy();
      expect(img.getAttribute('src')).toBe('favicon.ico');
    });
  });

  describe('Solution Link', () => {
    it('should display the solution link with call-to-action text', () => {
      const solutionLink = fixture.nativeElement.querySelector('a[href="https://github.com/jfaldanam/py_challenge_solution"]');
      expect(solutionLink).toBeTruthy();
      expect(solutionLink.textContent).toContain('Find the implementation here!');
    });

    it('should link to the solution GitHub repository', () => {
      const solutionLink = fixture.nativeElement.querySelector('a[href="https://github.com/jfaldanam/py_challenge_solution"]');
      expect(solutionLink.getAttribute('href')).toBe('https://github.com/jfaldanam/py_challenge_solution');
    });

    it('should contain a GitHub icon (SVG)', () => {
      const solutionLink = fixture.nativeElement.querySelector('a[href="https://github.com/jfaldanam/py_challenge_solution"]');
      const svg = solutionLink.querySelector('svg');
      expect(svg).toBeTruthy();
    });
  });

  describe('Accessibility', () => {
    it('should have alt text for the favicon image', () => {
      const img = fixture.nativeElement.querySelector('img');
      expect(img.getAttribute('alt')).toBeTruthy();
      expect(img.getAttribute('alt').length).toBeGreaterThan(0);
    });

    it('should have exactly two navigation links', () => {
      const links = fixture.nativeElement.querySelectorAll('a');
      expect(links.length).toBe(2);
    });

    it('should have valid href on all links', () => {
      const links = fixture.nativeElement.querySelectorAll('a');
      links.forEach((link: HTMLAnchorElement) => {
        const href = link.getAttribute('href');
        expect(href).toBeTruthy();
        expect(href).toMatch(/^https?:\/\//);
      });
    });
  });
});